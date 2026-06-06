import json
import re
from pathlib import Path
import statistics

sacred_dir = Path(__file__).parent / 'runs' / 'sacred'

epoch_pattern = re.compile(
    r'Epoch (\d+):.*?val_accuracy=([\d.]+),'
    r' val_fscore=([\d.]+), val_loss=([\d.]+),'
    r' val_precision=([\d.]+), val_recall=([\d.]+)'
)

results = []
for run_id in range(1, 11):
    run_dir = sacred_dir / str(run_id)
    if not run_dir.is_dir():
        continue

    config = json.load(open(run_dir / 'config.json'))
    fold = config.get('fold', '?')
    cout = (run_dir / 'cout.txt').read_text()
    status = json.load(open(run_dir / 'run.json')).get('status', '?')

    best_val_loss = float('inf')
    best_metrics = None
    best_epoch = None
    for line in cout.split('\n'):
        m = epoch_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            val_accuracy = float(m.group(2))
            val_fscore = float(m.group(3))
            val_loss = float(m.group(4))
            val_precision = float(m.group(5))
            val_recall = float(m.group(6))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = (val_accuracy, val_fscore, val_precision, val_recall)

    if best_metrics:
        results.append((fold, best_epoch, *best_metrics, best_val_loss, status))
        va, vf, vp, vr = best_metrics
        print(f'Fold {fold}: epoch={best_epoch:3d}  '
              f'acc={va:.4f}  f1={vf:.4f}  prec={vp:.4f}  rec={vr:.4f}  '
              f'val_loss={best_val_loss:.4f}  [{status}]')
    else:
        print(f'Fold {fold}: no metrics found  [{status}]')

if results:
    print()
    names = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    for i, name in enumerate(names):
        vals = [r[2+i] for r in results]
        mean = statistics.mean(vals) * 100
        stdev = statistics.stdev(vals) * 100 if len(vals) > 1 else 0
        print(f'{name}: {mean:.1f}% ± {stdev:.1f}%')
    print(f'\nPaper: 98.9% accuracy, 98.7% F1-score')
