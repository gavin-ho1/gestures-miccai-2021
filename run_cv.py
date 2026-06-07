#!/usr/bin/env python3
"""Run 10-fold CV sequentially (single process)."""
import subprocess
import sys
import time
import os
from pathlib import Path

BASE_CMD = [
    sys.executable, 'train_features_lstm.py', 'with',
    'gpus=0', 'args.gpus=0',
    'experiment_name=lstm_feats_jitter_4_agg_blstm_segs_16',
    'jitter_mode=4', 'aggregation=blstm', 'num_segments=16',
    'percentage_cores=100', 'batch_size=64',
]

ENV = {**os.environ, 'PYTHONUNBUFFERED': '1'}


def main():
    root = Path(__file__).parent
    os.chdir(root)

    print('=== MICCAI 2021 Gesture Classification — 10-fold CV ===')
    print(f'Using {os.cpu_count()} CPUs\n')

    fold_times = {}
    failed = []

    for fold in range(10):
        cmd = BASE_CMD + [f'fold={fold}']
        t0 = time.time()
        print(f'\n{"="*60}')
        print(f'Fold {fold + 1}/10 — starting')
        print(f'{"="*60}\n')
        result = subprocess.run(cmd, env=ENV)
        elapsed = time.time() - t0
        fold_times[fold] = elapsed
        status = 'OK' if result.returncode == 0 else 'FAILED'
        print(f'\nFold {fold + 1}: {status} ({elapsed/60:.1f} min)\n')
        if result.returncode != 0:
            failed.append(fold)

    print()
    print(f'Fold times: {fold_times}')
    if failed:
        print(f'FAILED folds: {failed}')
        sys.exit(1)

    print('\nAll folds completed!')
    print('Running evaluation...')
    result = subprocess.run(
        [sys.executable, 'evaluate_cv.py'],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr[-1000:])


if __name__ == '__main__':
    main()
