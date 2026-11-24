#!/usr/bin/env python3
"""Run missing SAE baseline ablations, one GPU per Docker, preserving -res legacy format."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ISOS', 'Bickley', 'Dibco', 'Einsiedeln', 'Palm', 'PHI', 'Salzinnes')
CASES = ('noover_noaug', 'over_noaug', 'over_aug')


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', default=','.join(DATASETS), help='Comma-separated dataset names')
    p.add_argument('--jobs', default=','.join(CASES), help='Comma-separated baseline names')
    p.add_argument('--lambdas', default='1,8,32', help='Comma-separated annotated patch counts')
    p.add_argument('--gpu', default='0', help='GPU visible inside this Docker')
    p.add_argument('--data-root', default='/workspace/datasets')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--np', type=int, default=1024, help='Number of patches per epoch with oversampling')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def subset(value, allowed, name):
    entries = [x.strip() for x in value.split(',') if x.strip()]
    if not entries or len(set(entries)) != len(entries) or any(x not in allowed for x in entries):
        raise SystemExit('Invalid %s %r; available: %s' % (name, value, ','.join(map(str, allowed))))
    return entries


def run(opts, dataset, case, lam):
    data = Path(opts.data_root) / dataset
    if case == 'noover_noaug':
        aug = ['none']
        npatches = lam
    elif case == 'over_noaug':
        aug = ['random']
        npatches = opts.np
    else:
        aug = ['random', 'flipH', 'flipV', 'rot', 'scale']
        npatches = opts.np

    outdir = ROOT / 'results' / 'sae_baselines_1page' / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    ident = '%s_lambda%d' % (case, lam)
    result_file = outdir / (ident + '_legacy_results.txt')
    marker = outdir / (ident + '.done')
    logtrain = ROOT / 'logs' / 'train' / ('out_CNN_%s_%s_nap%d.txt' % (dataset, case, lam))
    logtest = ROOT / 'logs' / 'test' / ('out_CNN_%s_%s_nap%d_--test.txt' % (dataset, case, lam))
    logtrain.parent.mkdir(parents=True, exist_ok=True)
    logtest.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists() and result_file.exists() and not opts.force:
        print('SKIP already completed:', dataset, ident, flush=True)
        return True

    common = [opts.python, '-u', 'main.py',
              '-db_train_src', str(data / 'train' / 'SRC'),
              '-db_train_gt', str(data / 'train' / 'GT'),
              '-db_test_src', str(data / 'test' / 'SRC'),
              '-db_test_gt', str(data / 'test' / 'GT'),
              '-aug', *aug, '-window_w', '256', '-window_h', '256',
              '-l', '4', '-f', '32', '-k', '3', '-drop', '0.2',
              '-pages_train', '1', '-npatches', str(npatches),
              '-n_annotated_patches', str(lam), '-ink_rate', '0.02',
              '-e', str(opts.epochs), '-b', '32', '-verbose', '1', '-gpu', '0']
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
    for stage, cmd, logfile in [('train', common, logtrain),
                                ('test', common + ['--test', '-res', str(result_file)], logtest)]:
        print('%s %s %s: %s' % (dataset, ident, stage, ' '.join(cmd)), flush=True)
        if opts.dry_run:
            continue
        if stage == 'train':
            marker.unlink(missing_ok=True)
            result_file.unlink(missing_ok=True)
        with logfile.open('w', encoding='utf-8') as stream:
            status = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        if status != 0:
            print('ERROR: %s %s failed (%s), see %s' % (dataset, ident, status, logfile), file=sys.stderr, flush=True)
            return False
    if not opts.dry_run:
        marker.write_text('ok\n', encoding='utf-8')
    return True


def main():
    opts = arguments()
    datasets = subset(opts.datasets, DATASETS, 'datasets')
    jobs = subset(opts.jobs, CASES, 'jobs')
    lambdas = [int(x) for x in subset(opts.lambdas, ('1','8','32'), 'lambdas')]
    failures = 0
    for dataset in datasets:
        if not opts.dry_run:
            data = Path(opts.data_root) / dataset
            for sub in ('train/SRC', 'train/GT', 'test/SRC', 'test/GT'):
                if not (data / sub).is_dir():
                    raise SystemExit('Dataset directory not found: %s' % (data / sub))
        for job in jobs:
            for lam in lambdas:
                if not run(opts, dataset, job, lam):
                    failures += 1
    print('Finished; failures:', failures, flush=True)
    return int(failures != 0)

if __name__ == '__main__':
    sys.exit(main())
