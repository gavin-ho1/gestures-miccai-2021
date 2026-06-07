#!/usr/bin/env python3
"""Run a 10-fold CV for one (seed, gamma) configuration, then (optionally) evaluate.

Examples:
  python run_cv.py                      # gamma=4, seed unset (non-deterministic)
  python run_cv.py --seed 0             # reproducible run, gamma=4
  python run_cv.py --seed 0 --gamma 2   # gamma=2 (Beta(2,2)) sampling
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUNS_DIR = Path(__file__).parent / 'runs'


def experiment_name(gamma, seed):
    name = f'lstm_feats_jitter_{gamma}_agg_blstm_segs_16'
    if seed is not None:
        name += f'_seed_{seed}'
    return name


def base_cmd(gamma, seed, exp_name):
    cmd = [
        sys.executable, 'train_features_lstm.py', 'with',
        'gpus=0', 'args.gpus=0',
        f'experiment_name={exp_name}',
        f'jitter_mode={gamma}', 'aggregation=blstm', 'num_segments=16',
        'percentage_cores=100', 'batch_size=64',
    ]
    if seed is not None:
        cmd.append(f'seed={seed}')
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixes the RNG for a reproducible run. Unset = Sacred picks one.')
    parser.add_argument('--gamma', default='4',
                        help='Beta concentration gamma for training-snippet sampling.')
    parser.add_argument('--no-eval', action='store_true',
                        help='Train only; skip the per-seizure evaluation at the end.')
    args = parser.parse_args()

    root = Path(__file__).parent
    os.chdir(root)
    exp_name = experiment_name(args.gamma, args.seed)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}

    # Start from a clean experiment dir so each fold ends with exactly one
    # checkpoint (prevents stale checkpoints from a previous run of the same
    # config being picked up by evaluate_cv.py).
    shutil.rmtree(RUNS_DIR / exp_name, ignore_errors=True)

    print(f'=== 10-fold CV  gamma={args.gamma}  seed={args.seed}  exp={exp_name} ===')
    print(f'Using {os.cpu_count()} CPUs\n')

    failed = []
    for fold in range(10):
        cmd = base_cmd(args.gamma, args.seed, exp_name) + [f'fold={fold}']
        t0 = time.time()
        print(f'\n{"="*60}')
        print(f'Fold {fold + 1}/10 — starting (gamma={args.gamma} seed={args.seed})')
        print(f'{"="*60}\n')
        result = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0
        status = 'OK' if result.returncode == 0 else 'FAILED'
        print(f'\nFold {fold + 1}: {status} ({elapsed/60:.1f} min)\n')
        if result.returncode != 0:
            failed.append(fold)

    if failed:
        print(f'FAILED folds: {failed}')
        sys.exit(1)

    print('\nAll folds completed!')
    if not args.no_eval:
        print('Running evaluation...\n')
        subprocess.run([sys.executable, 'evaluate_cv.py', exp_name], env=env)


if __name__ == '__main__':
    main()
