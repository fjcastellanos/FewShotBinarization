from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def _starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, max(1, length - patch + 1), stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_page_tiled(
    model,
    image_rgb: np.ndarray,
    patch_h: int,
    patch_w: int,
    batch_size: int = 4,
    overlap: int = 64,
    mixed_precision: bool = True,
) -> np.ndarray:
    device = next(model.parameters()).device
    stride_h = max(1, patch_h - overlap)
    stride_w = max(1, patch_w - overlap)
    ys = _starts(image_rgb.shape[0], patch_h, stride_h)
    xs = _starts(image_rgb.shape[1], patch_w, stride_w)

    prob_sum = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    count = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    patches = []
    coords = []

    def flush():
        nonlocal patches, coords
        if not patches:
            return
        x = torch.stack(patches).to(device)
        with torch.no_grad():
            if device.type == "cuda" and mixed_precision:
                dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
                ctx = torch.autocast("cuda", dtype=dtype)
            else:
                ctx = torch.autocast(device_type=device.type, enabled=False)
            with ctx:
                logits = model(x)
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
        probs = probs.cpu().numpy()
        for prob, (top, left, real_h, real_w) in zip(probs, coords):
            prob_sum[top:top + real_h, left:left + real_w] += prob[:real_h, :real_w]
            count[top:top + real_h, left:left + real_w] += 1.0
        patches, coords = [], []

    for top in ys:
        for left in xs:
            crop = image_rgb[top:top + patch_h, left:left + patch_w]
            real_h, real_w = crop.shape[:2]
            if real_h != patch_h or real_w != patch_w:
                padded = np.full((patch_h, patch_w, 3), 255, dtype=np.uint8)
                padded[:real_h, :real_w] = crop
                crop = padded
            tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float() / 255.0
            patches.append(tensor)
            coords.append((top, left, real_h, real_w))
            if len(patches) >= batch_size:
                flush()
    flush()
    return prob_sum / np.maximum(count, 1.0)


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    epsilon = 0.00001
    precision = float(tp / (tp + fp + epsilon))
    recall = float(tp / (tp + fn + epsilon))
    f1 = float(2.0 * precision * recall / (precision + recall + epsilon))
    return precision, recall, f1


def best_threshold(
    probabilities: Sequence[np.ndarray],
    gts: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray] | None = None,
    step: float = 0.05,
):
    if step <= 0 or step >= 1:
        raise ValueError("threshold step must be in (0,1)")
    thresholds = np.arange(step, 1.0, step)
    best = (-1.0, 0.5, 0.0, 0.0)
    for threshold in thresholds:
        preds = []
        refs = []
        for i, (prob, gt) in enumerate(zip(probabilities, gts)):
            if valid_masks is None:
                valid = np.ones_like(gt, dtype=bool)
            else:
                valid = valid_masks[i].astype(bool)
            # Legacy util.run_test() uses a strict > threshold comparison.
            preds.append(prob[valid] > threshold)
            refs.append(gt[valid].astype(bool))
        if not preds:
            continue
        precision, recall, f1 = precision_recall_f1(np.concatenate(preds), np.concatenate(refs))
        if f1 > best[0]:
            best = (f1, float(threshold), precision, recall)
    return best  # f1, threshold, precision, recall
