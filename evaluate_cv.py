"""
Paper-faithful evaluation (Sec 3, p.6):
  - For each fold, use the checkpoint saved by the training run (the epoch with
    the lowest validation loss; ModelCheckpoint keeps only that one).
  - Predict the TCS (positive) probability with the softmax head for BOTH video
    streams of each seizure, average them ("consensus probability"), and
    threshold at 0.5.
  - Pool every fold's per-seizure predictions into a single confusion matrix.
Discarded-FOV streams are excluded from training but USED for evaluation, so the
evaluation dataset is built with discard=False and its own cache file.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FeaturesSequencesDataset
from models import RecurrentModel, MeanModel
from training import get_fold_split

runs_dir = Path(__file__).parent / 'runs'
sacred_dir = runs_dir / 'sacred'
exp_dir = runs_dir / 'lstm_feats_jitter_4_agg_blstm_segs_16'
root_dir = Path(__file__).parent / 'dataset'
NUM_FOLDS = 10
MIN_DURATION = 15  # must match min_seizure_duration used for training


def get_latest_fold_configs():
    """Map each fold -> its most recent sacred config (later runs win)."""
    fold_config = {}
    if not sacred_dir.is_dir():
        return fold_config
    run_dirs = [d for d in sacred_dir.iterdir() if d.name.isdigit()]
    for run_dir in sorted(run_dirs, key=lambda p: int(p.name)):
        cfg_path = run_dir / 'config.json'
        if not cfg_path.is_file():
            continue
        try:
            cfg = json.load(open(cfg_path))
        except (json.JSONDecodeError, OSError):
            continue
        if 'fold' in cfg:
            fold_config[cfg['fold']] = cfg
    return fold_config


def build_model(cfg, state_dict):
    hidden = cfg.get('hidden_units', 64)
    aggregation = cfg.get('aggregation', 'blstm')
    if aggregation == 'mean':
        model = MeanModel()
    else:
        model = RecurrentModel(
            hidden_size=hidden,
            bidirectional=(aggregation == 'blstm'),
        )
    # state_dict keys carry a 'model.' prefix from the LightningModule wrapper.
    sd = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    fold_config = get_latest_fold_configs()

    all_true, all_preds, all_ssids = [], [], []

    for fold in range(NUM_FOLDS):
        fold_dir = exp_dir / f'fold_{fold}'
        ckpts = sorted(fold_dir.glob('*.ckpt')) if fold_dir.is_dir() else []
        if not ckpts:
            print(f'Fold {fold}: no checkpoint found in {fold_dir}')
            continue
        if len(ckpts) > 1:
            print(f'Fold {fold}: WARNING {len(ckpts)} checkpoints, using {ckpts[-1].name}')
        ckpt_path = ckpts[-1]
        cfg = fold_config.get(fold, {})
        num_segments = cfg.get('num_segments', 16)
        print(f'Fold {fold}: checkpoint={ckpt_path.name}  num_segments={num_segments}')

        state = torch.load(str(ckpt_path), map_location='cpu')
        model = build_model(cfg, state['state_dict'])

        _, val_ids, _ = get_fold_split(
            root_dir, k=fold, num_folds=NUM_FOLDS,
            num_holdout_folds=0, min_duration=MIN_DURATION,
        )
        dataset = FeaturesSequencesDataset(
            root_dir, frames_per_clip=8, frame_rate=15,
            subject_and_seizure_ids=val_ids,
            cache_path=Path('/tmp') / f'dataset_eval_fold{fold}.pth',
            num_segments=num_segments,
            jitter_mode='middle',  # gamma -> inf, central snippet
            discard=False,         # discarded-FOV streams are used for evaluation
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0,
        )

        # Collect per-view positive-class probabilities for this fold.
        seizure_preds = defaultdict(list)
        seizure_true = {}
        for batch in dataloader:
            with torch.no_grad():
                logits = model(batch['sequence'])              # [N, 2]
                probs = torch.softmax(logits, dim=1)[:, 1]      # P(TCS)
            for i in range(len(batch['pnt_szr_cam'])):
                ssid = '_'.join(batch['pnt_szr_cam'][i].split('_')[:2])
                seizure_preds[ssid].append(float(probs[i]))
                seizure_true[ssid] = int(batch['gtcs'][i])

        # Consensus probability per seizure = mean over its video streams.
        for ssid, preds in seizure_preds.items():
            all_preds.append(float(np.mean(preds)))
            all_true.append(seizure_true[ssid])
            all_ssids.append(ssid)

    if not all_true:
        print('No predictions collected. Did the training run complete?')
        return

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    pred_labels = (all_preds >= 0.5).astype(int)  # consensus thresholded at 0.5
    tp = int(((pred_labels == 1) & (all_true == 1)).sum())
    tn = int(((pred_labels == 0) & (all_true == 0)).sum())
    fp = int(((pred_labels == 1) & (all_true == 0)).sum())
    fn = int(((pred_labels == 0) & (all_true == 1)).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print()
    print(f'Pooled per-seizure evaluation over {len(all_true)} seizures '
          f'(unique: {len(set(all_ssids))})')
    print(f'TP={tp}  TN={tn}  FP={fp}  FN={fn}')
    print(f'Accuracy:  {accuracy*100:.1f}%')
    print(f'F1-score:  {f1*100:.1f}%')
    print(f'Precision: {precision*100:.1f}%')
    print(f'Recall:    {recall*100:.1f}%')
    print(f'\nPaper: 98.9% accuracy, 98.7% F1-score (77 TP, 104 TN, 2 FP, 0 FN)')


if __name__ == '__main__':
    main()
