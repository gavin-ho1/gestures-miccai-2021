#!/usr/bin/env python3
"""Sweep seeds (and optionally gamma) to characterise best vs mean performance.

The paper's 98.9% is the single best configuration of a sweep. A fair comparison
therefore runs several seeds (and optionally gamma in {2,4}) and reports both the
best and the mean +/- std of the pooled per-seizure accuracy / F1.

WARNING: each (seed, gamma) trains a full 10-fold CV. On CPU here that is roughly
3-4 hours PER configuration. Start small, e.g. `--seeds 0` to smoke-test, before
launching the full sweep.

Examples:
  python run_sweep.py --seeds 0                 # one reproducible run
  python run_sweep.py --seeds 0,1,2             # 3 seeds, gamma=4
  python run_sweep.py --seeds 0,1,2 --gammas 2,4
"""
import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_one(seed, gamma):
    exp = f'lstm_feats_jitter_{gamma}_agg_blstm_segs_16_seed_{seed}'
    # Train (no eval), then evaluate separately so we can capture a small output.
    train = subprocess.run(
        [sys.executable, 'run_cv.py', '--seed', str(seed), '--gamma', str(gamma), '--no-eval'],
        cwd=ROOT,
    )
    if train.returncode != 0:
        print(f'  training FAILED for seed={seed} gamma={gamma}')
        return None
    res = subprocess.run(
        [sys.executable, 'evaluate_cv.py', exp],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(res.stdout)
    acc = re.search(r'Accuracy:\s+([\d.]+)%', res.stdout)
    f1 = re.search(r'F1-score:\s+([\d.]+)%', res.stdout)
    if not (acc and f1):
        print(f'  could not parse result for seed={seed} gamma={gamma}')
        print(res.stderr[-500:])
        return None
    return float(acc.group(1)), float(f1.group(1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0,1,2', help='comma-separated seeds')
    p.add_argument('--gammas', default='4', help='comma-separated gammas, e.g. 2,4')
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    gammas = [g.strip() for g in args.gammas.split(',') if g.strip()]

    results = {g: [] for g in gammas}
    for gamma in gammas:
        for seed in seeds:
            print(f'\n########## gamma={gamma} seed={seed} ##########')
            r = run_one(seed, gamma)
            if r:
                results[gamma].append((seed, r[0], r[1]))

    print('\n' + '=' * 64)
    print('SWEEP SUMMARY  (pooled per-seizure accuracy / F1)')
    print('=' * 64)
    for gamma, rows in results.items():
        if not rows:
            print(f'\ngamma={gamma}: no successful runs')
            continue
        accs = [a for _, a, _ in rows]
        f1s = [f for _, _, f in rows]
        print(f'\ngamma={gamma}:')
        for seed, acc, f1 in rows:
            print(f'  seed={seed:>3}:  acc={acc:5.1f}%   f1={f1:5.1f}%')
        best = max(rows, key=lambda r: r[2])  # best by F1
        print(f'  BEST   :  acc={best[1]:5.1f}%   f1={best[2]:5.1f}%   (seed={best[0]})')
        if len(accs) > 1:
            print(f'  MEAN   :  acc={statistics.mean(accs):.1f}+/-{statistics.stdev(accs):.1f}%'
                  f'   f1={statistics.mean(f1s):.1f}+/-{statistics.stdev(f1s):.1f}%')
        else:
            print(f'  MEAN   :  acc={statistics.mean(accs):.1f}%   f1={statistics.mean(f1s):.1f}%')
    print('\nPaper best: 98.9% accuracy / 98.7% F1-score')


if __name__ == '__main__':
    main()
