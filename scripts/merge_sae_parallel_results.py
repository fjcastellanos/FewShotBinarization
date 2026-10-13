#!/usr/bin/env python3
"""Merge per-experiment legacy -res files copied from one or more computers."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('dirs', nargs='+', help='Directories containing *_legacy_results.txt')
p.add_argument('--output', default='results/sae_merged')
a = p.parse_args()
records = {}
for directory in a.dirs:
    root = Path(directory)
    for file in root.rglob('*_legacy_results.txt'):
        stem = file.name.removesuffix('_legacy_results.txt')
        if '_lambda' not in stem:
            continue
        method, num = stem.rsplit('_lambda', 1)
        if method not in ('cutmix', 'mixup') or num not in ('1', '8', '32'):
            continue
        key = (method, int(num))
        data = file.read_text(encoding='utf8').splitlines()
        test_rows = [line for line in data if 'TEST' in line.split(';')]
        if not test_rows:
            raise ValueError(f'Missing TEST record: {file}')
        row = test_rows[-1]
        bits = row.split(';')
        f1 = float(bits[bits.index('TEST') + 1].replace(',', '.'))
        if key in records:
            raise ValueError(f'Duplicate experiment {key}: {records[key][2]} and {file}')
        records[key] = (f1, row, file)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
with (out / 'results.csv').open('w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['method', 'lambda', 'f1_percent', 'source'])
    for (method, n), (f1, row, file) in sorted(records.items()):
        writer.writerow((method, n, f1, file))
with (out / 'legacy_results_combined.txt').open('w', encoding='utf8') as f:
    for key in sorted(records):
        f.write(records[key][1].rstrip('\n') + '\n')
with (out / 'table_rows.tex').open('w', encoding='utf8') as f:
    for method, label in [('cutmix', 'Over. + CutMix'), ('mixup', 'Over. + MixUp')]:
        values = [f'{records[(method,n)][0]:.1f}' if (method,n) in records else '' for n in (1,8,32)]
        f.write(label + ' & ' + ' & '.join(values) + ' ' + chr(92)*2 + chr(10))
print('Found', len(records), 'of 6 experiments; wrote', out)
