from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Sequence, Tuple

import cv2
import numpy as np

from .data import build_legacy_annotation_mask


LEGACY_RESULT_HEADER = (
    "Train;Test;PAG;Num pages train;ANN;Num annotations per page;"
    "PAT;Num random patches;Ink rate;VAL;Th_bin;F1-val;P-val;R-val;"
    "Num annotated patches-val;Maximum num annotated patches-val;TEST;"
    "F1-test;P-test;R-test;Num annotated patches-test;"
    "Maximum num annotated patches-test;IoU-test;Specificity-test;"
    "TP-test;TN-test;FP-test;FN-test;TimePerPage(s);"
)


def sanitize_token(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in text)


def infer_dataset_name(train_src: str) -> str:
    """Infer the dataset folder from paths such as datasets/Dibco/train/SRC."""
    path = Path(train_src)
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "train" in lowered:
        idx = lowered.index("train")
        if idx > 0:
            return sanitize_token(parts[idx - 1])
    if path.name.lower() in {"src", "images", "image"} and path.parent.name:
        parent = path.parent
        if parent.name.lower() in {"train", "training"} and parent.parent.name:
            return sanitize_token(parent.parent.name)
    return sanitize_token(path.parent.name or path.name)


def legacy_number_to_string(number) -> str:
    """Match main.py's historical percentage formatting, including its quirks."""
    return str(round(float(number) * 100.0, 1)).replace(".", ",")


def legacy_result_line(
    *,
    db_train_src: str,
    db_test_src: str,
    pages_train: int,
    n_annotated_patches: int,
    npatches: int,
    ink_rate: float,
    threshold: float,
    val_f1: float,
    val_precision: float,
    val_recall: float,
    n_annotated_val: int,
    max_annotated_val: int,
    test_metrics: Dict,
    n_annotated_train: int,
    max_annotated_train: int,
    avg_elapsed: float,
) -> str:
    """Produce the same semicolon-separated row layout as legacy main.py."""
    sep = ";"
    props = str(db_train_src) + sep
    props += str(db_test_src) + sep
    props += "PAG" + sep
    props += str(pages_train) + sep
    props += "ANN" + sep
    props += str(n_annotated_patches) + sep
    props += "PAT" + sep
    props += str(npatches) + sep
    props += str(ink_rate).replace(".", ",") + sep

    result = props + sep
    result += "VAL" + sep
    result += str(threshold).replace(".", ",") + sep
    result += legacy_number_to_string(val_f1) + sep
    result += legacy_number_to_string(val_precision) + sep
    result += legacy_number_to_string(val_recall) + sep
    result += str(n_annotated_val) + sep
    result += str(max_annotated_val) + sep

    result += sep + "TEST" + sep
    result += legacy_number_to_string(test_metrics["f1"]) + sep
    result += legacy_number_to_string(test_metrics["precision"]) + sep
    result += legacy_number_to_string(test_metrics["recall"]) + sep
    # Historical main.py labels these columns as test, but actually stores the
    # counts from the selected training pages. Preserve that behavior so rows
    # remain directly comparable with the existing legacy result files.
    result += str(n_annotated_train) + sep
    result += str(max_annotated_train) + sep
    result += legacy_number_to_string(test_metrics["iou"]) + sep
    result += legacy_number_to_string(test_metrics["specificity"]) + sep
    result += legacy_number_to_string(test_metrics["tp"]) + sep
    result += legacy_number_to_string(test_metrics["tn"]) + sep
    result += legacy_number_to_string(test_metrics["fp"]) + sep
    result += legacy_number_to_string(test_metrics["fn"]) + sep
    result += str(float(avg_elapsed)).replace(".", ",") + sep
    return result


def append_line(path: str, content: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(content + "\n")


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    gt = np.asarray(gt, dtype=bool).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")

    not_pred = np.logical_not(pred)
    not_gt = np.logical_not(gt)
    tp = int(np.logical_and(pred, gt).sum(dtype=np.int64))
    tn = int(np.logical_and(not_pred, not_gt).sum(dtype=np.int64))
    fp = int(np.logical_and(pred, not_gt).sum(dtype=np.int64))
    fn = int(np.logical_and(not_pred, gt).sum(dtype=np.int64))

    epsilon = 0.00001
    precision = float(tp / (tp + fp + epsilon))
    recall = float(tp / (tp + fn + epsilon))
    f1 = float(2.0 * precision * recall / (precision + recall + epsilon))
    specificity = float(tn / (tn + fp + epsilon))
    iou = float(tp / (tp + fp + fn + epsilon))
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def count_annotation_windows(
    pairs: Sequence[Tuple[str, str]],
    load_gt,
    patch_h: int,
    patch_w: int,
    n_annotated_patches: int,
    ink_rate: float,
) -> int:
    total = 0
    for _, gt_path in pairs:
        gt = load_gt(gt_path)
        _, windows = build_legacy_annotation_mask(
            gt,
            patch_h,
            patch_w,
            n_annotated_patches,
            ink_rate,
        )
        total += len(windows)
    return total


def _save_image(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if rgb and array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Could not save image: {path}")


def checkpoint_artifact_root(checkpoint: str, kind: str, model: str, experiment: str) -> Path:
    stem = sanitize_token(Path(checkpoint).stem)
    return Path(kind) / "modern" / sanitize_token(model) / sanitize_token(experiment) / stem


def page_output_base(
    checkpoint: str,
    kind: str,
    model: str,
    experiment: str,
    dataset_name: str,
    split: str,
    src_root: str,
    src_path: str,
) -> Path:
    root = checkpoint_artifact_root(checkpoint, kind, model, experiment)
    src_root_path = Path(src_root)
    src = Path(src_path)
    try:
        rel = src.relative_to(src_root_path)
    except ValueError:
        rel = Path(src.name)
    # Legacy layout effectively keeps: <dataset>/<split>/SRC/<filename>.
    return root / sanitize_token(dataset_name) / split / "SRC" / rel


def save_legacy_page_images(
    *,
    checkpoint: str,
    kind: str,
    model: str,
    experiment: str,
    dataset_name: str,
    split: str,
    src_root: str,
    src_path: str,
    image_rgb: np.ndarray,
    gt: np.ndarray,
    region_mask: np.ndarray,
    probability: np.ndarray | None = None,
    threshold: float | None = None,
) -> Dict[str, str]:
    """Save images with the same suffixes and pixel conventions as legacy util.py."""
    base = page_output_base(
        checkpoint,
        kind,
        model,
        experiment,
        dataset_name,
        split,
        src_root,
        src_path,
    )
    outputs: Dict[str, str] = {}

    gt_path = Path(str(base) + "_gt.png")
    gr_path = Path(str(base) + "_gr.png")
    region_path = Path(str(base) + "_annotated_regions.png")

    _save_image(gt_path, np.asarray(gt, dtype=np.uint8) * 255)
    # Legacy normalize_image() is (255-img)/255, and then _gr.png writes *255.
    # Therefore the historical result image is the inverted source image.
    _save_image(gr_path, 255 - np.asarray(image_rgb, dtype=np.uint8), rgb=True)
    _save_image(region_path, np.asarray(region_mask, dtype=np.uint8) * 255)
    outputs.update(gt=str(gt_path), gr=str(gr_path), annotated_regions=str(region_path))

    if probability is not None:
        pred_path = Path(str(base) + "_pred.png")
        pred_th_path = Path(str(base) + "_pred_th.png")
        probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        _save_image(pred_path, probability * 255.0)
        if threshold is None:
            raise ValueError("threshold is required when probability is provided")
        # Legacy run_test() uses strict > threshold.
        _save_image(pred_th_path, probability > float(threshold))
        outputs.update(pred=str(pred_path), pred_th=str(pred_th_path))

    return outputs
