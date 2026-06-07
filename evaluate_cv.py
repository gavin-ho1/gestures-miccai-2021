"""
Paper-faithful evaluation (Sec 3, p.6):
  - For each fold, load the checkpoint saved by the training run (the epoch with
    the lowest validation loss).
  - Predict the TCS (positive) probability with the softmax head for BOTH video
    streams of each seizure, average them, threshold at 0.5.
  - Pool every fold's per-seizure predictions into a single confusion matrix.
Discarded-FOV streams are excluded from training but USED for evaluation, so the
evaluation dataset is built with discard=False and its own cache file.

Usage:  python evaluate_cv.py [experiment_name]
        (defaults to the gamma=4, seed-unset experiment)
"""
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FeaturesSequencesDataset
from models import RecurrentModel
from training import get_fold_split

runs_dir = Path(__file__).parent / 'runs'
root_dir = Path(__file__).parent / 'dataset'

NUM_FOLDS = 10
MIN_DURATION = 15      # must match min_seizure_duration used for training
NUM_SEGMENTS = 16      # fixed architecture for this reproduction
HIDDEN = 64
BIDIRECTIONAL = True


def load_model(ckpt_path):
    state = torch.load(str(ckpt_path), map_location='cpu')
    model = RecurrentModel(hidden_size=HIDDEN, bidirectional=BIDIRECTIONAL)
    # state_dict keys carry a 'model.' prefix from the LightningModule wrapper.
    sd = {k.replace('model.', '', 1): v for k, v in state['state_dict'].items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_metrics(true, preds):
    true = np.asarray(true)
    preds = np.asarray(preds)
    lab = (preds >= 0.5).astype(int)  # consensus thresholded at 0.5
    tp = int(((lab == 1) & (true == 1)).sum())
    tn = int(((lab == 0) & (true == 0)).sum())
    fp = int(((lab == 1) & (true == 0)).sum())
    fn = int(((lab == 0) & (true == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n=n, acc=acc, prec=prec, rec=rec, f1=f1)


def pick_checkpoint(fold_dir):
    ckpts = list(fold_dir.glob('*.ckpt')) if fold_dir.is_dir() else []
    if not ckpts:
        return None
    # If more than one exists (e.g. a stale file), pick the highest epoch number.
    def epoch_of(p):
        digits = ''.join(filter(str.isdigit, p.stem))
        return int(digits) if digits else -1
    return sorted(ckpts, key=epoch_of)[-1]


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else 'lstm_feats_jitter_4_agg_blstm_segs_16'
    exp_dir = runs_dir / exp_name

    all_true, all_preds = [], []
    print(f'Experiment: {exp_name}\n')
    print(f'{"fold":>4}  {"checkpoint":>16}  {"n":>3}  {"acc":>6}  {"f1":>6}  TP/TN/FP/FN')
    for fold in range(NUM_FOLDS):
        ckpt_path = pick_checkpoint(exp_dir / f'fold_{fold}')
        if ckpt_path is None:
            print(f'{fold:>4}  (no checkpoint found)')
            continue
        model = load_model(ckpt_path)

        _, val_ids, _ = get_fold_split(
            root_dir, k=fold, num_folds=NUM_FOLDS,
            num_holdout_folds=0, min_duration=MIN_DURATION,
        )
        dataset = FeaturesSequencesDataset(
            root_dir, frames_per_clip=8, frame_rate=15,
            subject_and_seizure_ids=val_ids,
            cache_path=Path('/tmp') / f'dataset_eval_fold{fold}.pth',
            num_segments=NUM_SEGMENTS, jitter_mode='middle', discard=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=0,
        )

        seizure_preds = defaultdict(list)
        seizure_true = {}
        for batch in loader:
            with torch.no_grad():
                probs = torch.softmax(model(batch['sequence']), dim=1)[:, 1]
            for i in range(len(batch['pnt_szr_cam'])):
                ssid = '_'.join(batch['pnt_szr_cam'][i].split('_')[:2])
                seizure_preds[ssid].append(float(probs[i]))
                seizure_true[ssid] = int(batch['gtcs'][i])

        fold_true, fold_preds = [], []
        for ssid, ps in seizure_preds.items():
            fold_preds.append(float(np.mean(ps)))   # consensus over streams
            fold_true.append(seizure_true[ssid])
        all_true.extend(fold_true)
        all_preds.extend(fold_preds)

        m = compute_metrics(fold_true, fold_preds)
        print(f'{fold:>4}  {ckpt_path.name:>16}  {m["n"]:>3}  '
              f'{m["acc"]*100:>5.1f}%  {m["f1"]*100:>5.1f}%  '
              f'{m["tp"]}/{m["tn"]}/{m["fp"]}/{m["fn"]}')

    if not all_true:
        print('\nNo predictions collected. Did the training run complete?')
        return

    M = compute_metrics(all_true, all_preds)
    print(f'\nPooled per-seizure evaluation over {M["n"]} seizures')
    print(f'TP={M["tp"]}  TN={M["tn"]}  FP={M["fp"]}  FN={M["fn"]}')
    print(f'Accuracy:  {M["acc"]*100:.1f}%')
    print(f'F1-score:  {M["f1"]*100:.1f}%')
    print(f'Precision: {M["prec"]*100:.1f}%')
    print(f'Recall:    {M["rec"]*100:.1f}%')
    print(f'\nPaper: 98.9% accuracy, 98.7% F1-score (77 TP, 104 TN, 2 FP, 0 FN)')


if __name__ == '__main__':
    main()
