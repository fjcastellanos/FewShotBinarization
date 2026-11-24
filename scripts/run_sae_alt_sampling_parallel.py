#!/usr/bin/env python3
"""Run the SAE centered_context / output_masking experiments on several GPUs at once.

One worker per GPU pulls the next pending (dataset, experiment, lambda) from a
shared queue, so no GPU stays idle. Every task runs exactly the same train +
test commands as scripts/run_sae_table_baselines.py (same logs, same
-res result files, same .done markers), so finished tasks are skipped on
restart and results stay next to the over_noaug baseline.
"""
import argparse
import copy
import csv
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sae_table_baselines as table  # noqa: E402

ROOT = table.ROOT


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1', help='Comma-separated GPU ids; one job per GPU at a time')
    p.add_argument('--datasets', default=','.join(table.DATASETS), help='Comma-separated dataset names')
    p.add_argument('--jobs', default=','.join(table.ALT_CASES), help='centered_context,output_masking')
    p.add_argument('--lambdas', default='1,8,32', help='Comma-separated annotated patch counts')
    p.add_argument('--data-root', default='/workspace/datasets')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--np', type=int, default=1024, help='Number of patches per epoch (same as oversampling)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def test_f1(path):
    """F1-test (percent) from a legacy -res file, or '' if unavailable."""
    try:
        for line in reversed(Path(path).read_text(encoding='utf-8').splitlines()):
            parts = line.split(';')
            if 'TEST' in parts and parts.index('TEST') + 1 < len(parts):
                return parts[parts.index('TEST') + 1]
    except OSError:
        pass
    return ''


def write_summary(datasets, lambdas, jobs):
    path = ROOT / 'results' / 'sae_baselines_1page' / 'alt_sampling_summary.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = ['over_noaug'] + list(jobs)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['dataset', 'lambda'] + ['F1-test ' + c for c in cases])
        for dataset in datasets:
            for lam in lambdas:
                base = ROOT / 'results' / 'sae_baselines_1page' / dataset
                writer.writerow([dataset, lam] + [test_f1(base / ('%s_lambda%d_legacy_results.txt' % (c, lam)))
                                                  for c in cases])
    print('Summary (F1-test, compared with over_noaug):', path, flush=True)


def main():
    opts = arguments()
    gpus = [g.strip() for g in opts.gpus.split(',') if g.strip()]
    if not gpus or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        raise SystemExit('Expected unique GPU indices, e.g. --gpus 0,1')
    datasets = table.subset(opts.datasets, table.DATASETS, 'datasets')
    jobs = table.subset(opts.jobs, table.ALT_CASES, 'jobs')
    lambdas = [int(x) for x in table.subset(opts.lambdas, ('1', '8', '32'), 'lambdas')]
    if not opts.dry_run:
        for dataset in datasets:
            for sub in ('train/SRC', 'train/GT', 'test/SRC', 'test/GT'):
                if not (Path(opts.data_root) / dataset / sub).is_dir():
                    raise SystemExit('Dataset directory not found: %s' % (Path(opts.data_root) / dataset / sub))

    tasks = queue.Queue()
    for dataset in datasets:
        for lam in lambdas:
            for job in jobs:
                tasks.put((dataset, job, lam))
    total = tasks.qsize()
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        worker_opts = copy.copy(opts)
        worker_opts.gpu = gpu  # run() sets CUDA_VISIBLE_DEVICES from this
        while True:
            try:
                dataset, job, lam = tasks.get_nowait()
            except queue.Empty:
                return
            with lock:
                print('[GPU %s] START %s %s lambda=%d (%d pending)' % (gpu, dataset, job, lam, tasks.qsize()), flush=True)
            try:
                ok = table.run(worker_opts, dataset, job, lam)
            except Exception as ex:  # keep the other GPU working
                print('[GPU %s] ERROR %s %s lambda=%d: %s' % (gpu, dataset, job, lam, ex), file=sys.stderr, flush=True)
                ok = False
            with lock:
                if not ok:
                    failures.append('%s %s lambda=%d' % (dataset, job, lam))
                print('[GPU %s] %s %s %s lambda=%d' % (gpu, 'DONE' if ok else 'FAILED', dataset, job, lam), flush=True)

    print('Running %d tasks on GPUs %s' % (total, ','.join(gpus)), flush=True)
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not opts.dry_run:
        write_summary(datasets, lambdas, jobs)
    print('Finished %d tasks; failures: %d' % (total, len(failures)), flush=True)
    for failure in failures:
        print('FAILED:', failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
