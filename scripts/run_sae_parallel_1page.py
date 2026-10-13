#!/usr/bin/env python3
"""Parallel, restartable SAE Keras MixUp/CutMix experiments. Python stdlib only."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
ALL = [f'{method}:{n}' for method in ('cutmix', 'mixup') for n in (1, 8, 32)]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', default=os.getenv('SAE_JOBS', 'all'), help='Comma/space separated cutmix:1,mixup:8 or all')
    p.add_argument('--gpus', default=os.getenv('SAE_GPUS', '0,1'), help='GPU IDs visible on THIS computer')
    p.add_argument('--dataset', default=os.getenv('DATASET', 'ISOS'))
    p.add_argument('--data-root', default=os.getenv('DATA_ROOT', 'datasets'))
    p.add_argument('--results-root', default=os.getenv('SAE_RESULTS_ROOT', 'results/sae_mix_one_page'))
    p.add_argument('--np', type=int, default=int(os.getenv('NPATCHES', '1024')))
    p.add_argument('--epochs', type=int, default=int(os.getenv('EPOCHS', '200')))
    p.add_argument('--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', '32')))
    p.add_argument('--ink-rate', type=float, default=float(os.getenv('INK_RATE', '0.02')))
    p.add_argument('--mixup-min', type=float, default=float(os.getenv('MIXUP_MIN', '0.25')))
    p.add_argument('--mixup-max', type=float, default=float(os.getenv('MIXUP_MAX', '0.75')))
    p.add_argument('--python', default=os.getenv('PYTHON', sys.executable))
    p.add_argument('--host-label', default=os.getenv('SAE_HOST_LABEL', ''), help='Optional name distinguishing machines in summary')
    p.add_argument('--force', action='store_true', help='Re-run completed experiments')
    p.add_argument('--dry-run', action='store_true', help='Show commands without running')
    return p


def parse_jobs(value):
    jobs = ALL if value.strip().lower() == 'all' else [j.lower() for j in re.split(r'[\s,]+', value.strip()) if j]
    if not jobs or len(jobs) != len(set(jobs)) or any(j not in ALL for j in jobs):
        raise ValueError(f'Invalid/duplicate jobs: {jobs}; allowed: {", ".join(ALL)}')
    return jobs


def score_from_legacy(path):
    """Returns F1 TEST in percent using main.py's original semicolon output."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    for line in reversed(lines):
        parts = line.split(';')
        if 'TEST' in parts:
            idx = parts.index('TEST')
            if idx + 1 < len(parts):
                return float(parts[idx + 1].replace(',', '.'))
    raise ValueError(f'Cannot locate TEST F1 in {path}')


def run_one(job, gpu, opts):
    method, n = job.split(':')
    dataset = opts.dataset
    outdir = (ROOT / opts.results_root / dataset).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / f'{method}_lambda{n}'
    legacy_result = Path(str(base) + '_legacy_results.txt')
    done = Path(str(base) + '.done')
    test_log = Path(str(base) + '_test.log')
    if done.is_file() and legacy_result.is_file() and not opts.force:
        score = score_from_legacy(legacy_result)
        return method, n, score, str(legacy_result), str(test_log), 'skipped'

    src = (ROOT / opts.data_root / dataset).resolve()
    serial = 'random.flipH.flipV.rot.scale'
    # Same convention for the console logs as the historical sh_train/test scripts.
    stem = (f'out_CNN_inkth_{opts.ink_rate}_{dataset}_au g{serial}'
            f'_w256_h256_l4_f32_k3_d0.2_pt1_np{opts.np}_nap{n}'
            f'_e{opts.epochs}_b{opts.batch_size}_{method}').replace('au g', 'aug')
    logs_train = ROOT / 'logs' / 'train' / (stem + '.txt')
    logs_test = ROOT / 'logs' / 'test' / (stem + '_--test.txt')
    logs_train.parent.mkdir(parents=True, exist_ok=True)
    logs_test.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Isolate TensorFlow GPU per process. main.py sees only one physical GPU, index 0.
    env['CUDA_VISIBLE_DEVICES'] = gpu
    common = [opts.python, '-u', 'main.py',
        '-db_train_src', str(src / 'train/SRC'), '-db_train_gt', str(src / 'train/GT'),
        '-db_test_src', str(src / 'test/SRC'), '-db_test_gt', str(src / 'test/GT'),
        '-aug', 'random', 'flipH', 'flipV', 'rot', 'scale',
        '-window_w', '256', '-window_h', '256', '-l', '4', '-f', '32', '-k', '3',
        '-drop', '0.2', '-pages_train', '1', '-npatches', str(opts.np),
        '-n_annotated_patches', n, '-ink_rate', str(opts.ink_rate),
        '-e', str(opts.epochs), '-b', str(opts.batch_size), '-verbose', '1',
        '-gpu', '0', '--sae_mix', method, '--mixup_min', str(opts.mixup_min),
        '--mixup_max', str(opts.mixup_max)]
    # Preserve historical -res format, but per-job rather than writing concurrently to a shared file.
    for suffix, cmd, logfile in [('train', common, logs_train),
                                  ('test', common + ['--test', '-res', str(legacy_result)], logs_test)]:
        print(f'[{gpu}] {job} {suffix}: {" ".join(map(str, cmd))}', flush=True)
        if opts.dry_run:
            continue
        if suffix == 'train':
            # Do not append results from a stale interrupted run.
            done.unlink(missing_ok=True)
            legacy_result.unlink(missing_ok=True)
        with logfile.open('w', encoding='utf-8') as stream:
            result = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{job} {suffix} failed (exit {result.returncode}). Read {logfile}')
    if opts.dry_run:
        return method, int(n), '', str(legacy_result), str(logs_test), 'dry-run'
    score = score_from_legacy(legacy_result)
    done.write_text('ok\n', encoding='utf-8')
    return method, int(n), score, str(legacy_result), str(logs_test), 'ok'


def write_summary(outdir, rows):
    path = outdir / 'results_parallel.csv'
    # CSV is additional; original -res semicolon records remain unchanged.
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method', 'lambda', 'f1_percent', 'legacy_results', 'test_log', 'status'])
        writer.writerows(sorted(rows, key=lambda x: (x[0], x[1])))
    print('Summary:', path)


def main():
    opts = make_parser().parse_args()
    jobs = parse_jobs(opts.jobs)
    gpus = [v for v in re.split(r'[\s,]+', opts.gpus) if v]
    if not gpus or len(set(gpus)) != len(gpus) or not all(x.isdigit() for x in gpus):
        raise ValueError('Expected unique GPU indices, e.g. --gpus 0,1')
    src = (ROOT / opts.data_root / opts.dataset).resolve()
    if not opts.dry_run:
        for subpath in ('train/SRC', 'train/GT', 'test/SRC', 'test/GT'):
            if not (src / subpath).is_dir():
                raise ValueError(f'Missing dataset directory: {src / subpath}')
    rows = []
    # One worker/GPU, with jobs assigned round-robin and serial within each worker.
    queues = [jobs[i::len(gpus)] for i in range(len(gpus))]
    def run_queue(gpu, queue):
        records = []
        for job in queue:
            records.append(run_one(job, gpu, opts))
        return records
    errors = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(run_queue, gpu, queue) for gpu, queue in zip(gpus, queues) if queue]
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as ex:
                errors.append(str(ex))
    write_summary((ROOT / opts.results_root / opts.dataset).resolve(), rows)
    for error in errors:
        print('ERROR:', error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
