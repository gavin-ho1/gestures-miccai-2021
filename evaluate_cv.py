import json
import re
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FeaturesSequencesDataset
from models import RecurrentModel
from training import get_fold_split
from utils import torch_load

runs_dir = Path(__file__).parent / 'runs'
sacred_dir = runs_dir / 'sacred'
exp_dir = runs_dir / 'lstm_feats_jitter_4_agg_blstm_segs_16'
root_dir = Path(__file__).parent / 'dataset'

epoch_pattern = re.compile(
    r'Epoch (\d+):.*?val_accuracy=([\d.]+),'
    r' val_fscore=([\d.]+), val_loss=([\d.]+),'
    r' val_precision=([\d.]+), val_recall=([\d.]+)'
)

def find_best_epoch(run_dir):
    cout = (run_dir / 'cout.txt').read_text()
    best_val_loss = float('inf')
    best_epoch = None
    for line in cout.split('\n'):
        m = epoch_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            val_loss = float(m.group(4))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
    return best_epoch

def get_checkpoint_for_epoch(epoch, fold):
    fold_dir = exp_dir / f'fold_{fold}'
    if not fold_dir.is_dir():
        return None
    ckpts = sorted(fold_dir.glob('*.ckpt'))
    for ckpt in ckpts:
        m = re.search(r'(\d+)', ckpt.stem)
        if m and int(m.group(1)) == epoch - 1:
            return ckpt
    return None

all_true = []
all_preds = []
all_ssids = []

for run_id in range(1, 11):
    run_dir = sacred_dir / str(run_id)
    if not run_dir.is_dir():
        continue
    config = json.load(open(run_dir / 'config.json'))
    fold = config['fold']
    best_epoch = find_best_epoch(run_dir)
    if best_epoch is None:
        print(f'Fold {fold}: no best epoch found')
        continue
    ckpt_path = get_checkpoint_for_epoch(best_epoch, fold)
    if ckpt_path is None:
        print(f'Fold {fold}: no checkpoint for epoch {best_epoch}')
        print(f'  Available: {[p.name for p in exp_dir.iterdir() if p.suffix == ".ckpt"]}')
        continue
    print(f'Fold {fold}: best epoch={best_epoch}, checkpoint={ckpt_path.name}')
    state = torch_load(str(ckpt_path), map_location='cpu')
    model = RecurrentModel(hidden_size=64, bidirectional=True)
    # state_dict keys have 'model.' prefix from pl.LightningModule wrapper
    sd = {k.replace('model.', '', 1): v for k, v in state['state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    _, val_ids, _ = get_fold_split(
        root_dir, k=fold, num_folds=10, num_holdout_folds=0, min_duration=15,
    )
    dataset = FeaturesSequencesDataset(
        root_dir, frames_per_clip=8, frame_rate=15,
        subject_and_seizure_ids=val_ids,
        cache_path=Path('/tmp') / f'dataset_val_fold{fold}.pth',
        num_segments=config['num_segments'],
        jitter_mode='middle',
        discard=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )
    fold_preds = []
    fold_true = []
    fold_ssids = []
    for batch in dataloader:
        with torch.no_grad():
            logits = model(batch['sequence'])
            probs = torch.sigmoid(logits).view(-1)
        for i in range(len(batch['pnt_szr_cam'])):
            ssid = '_'.join(batch['pnt_szr_cam'][i].split('_')[:2])
            fold_true.append(int(batch['gtcs'][i]))
            fold_preds.append(float(probs[i]))
            fold_ssids.append(ssid)
    # Average predictions per seizure (both views)
    seizure_preds = defaultdict(list)
    seizure_true = {}
    for ssid, pred, t in zip(fold_ssids, fold_preds, fold_true):
        seizure_preds[ssid].append(pred)
        seizure_true[ssid] = t
    for ssid, preds in seizure_preds.items():
        avg_pred = np.mean(preds)
        all_true.append(seizure_true[ssid])
        all_preds.append(avg_pred)
        all_ssids.append(ssid)

print()
all_preds = np.array(all_preds)
all_true = np.array(all_true)
pred_labels = (all_preds >= 0.5).astype(int)
tp = ((pred_labels == 1) & (all_true == 1)).sum()
tn = ((pred_labels == 0) & (all_true == 0)).sum()
fp = ((pred_labels == 1) & (all_true == 0)).sum()
fn = ((pred_labels == 0) & (all_true == 1)).sum()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
n_total = len(all_true)
n_unique = len(set(all_ssids))
print(f'Total predictions: {n_total} (unique seizures: {n_unique})')
print(f'TP={tp}  TN={tn}  FP={fp}  FN={fn}')
print(f'Accuracy:  {accuracy*100:.1f}%')
print(f'F1-score:  {f1*100:.1f}%')
print(f'Precision: {precision*100:.1f}%')
print(f'Recall:    {recall*100:.1f}%')
print(f'\nPaper: 98.9% accuracy, 98.7% F1-score')
