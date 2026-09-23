from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .inference import best_threshold, predict_page_tiled
from .losses import partial_cross_entropy


@dataclass
class TrainConfig:
    # Pretrained models + one annotated page: use a conservative ceiling and
    # let full-page validation F1 stop training before overfitting.
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 6e-5
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    grad_accum_steps: int = 8
    mixed_precision: bool = True
    num_workers: int = 0
    seed: int = 1
    threshold_step: float = 0.1

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * max(1, self.grad_accum_steps)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, cfg: TrainConfig):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _cuda_bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker is not None and checker())


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if _cuda_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler changed signature across PyTorch releases. Keep the
    # modern pipeline usable with the declared torch>=2.2 requirement.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def evaluate_full_pages(
    model,
    val_pages: Sequence,
    patch_h: int,
    patch_w: int,
    batch_size: int,
    overlap: int,
    mixed_precision: bool,
    threshold_step: float,
):
    """Evaluate complete validation page(s) and choose threshold on full GT."""
    probabilities = [
        predict_page_tiled(
            model,
            page.image,
            patch_h,
            patch_w,
            batch_size=batch_size,
            overlap=overlap,
            mixed_precision=mixed_precision,
        )
        for page in val_pages
    ]
    f1, threshold, precision, recall = best_threshold(
        probabilities,
        [page.full_gt for page in val_pages],
        valid_masks=None,
        step=threshold_step,
    )
    return {
        "f1": float(f1),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
    }


def _write_history(path: Path, history, cfg: TrainConfig, metadata: Dict):
    payload = {
        "train_config": asdict(cfg),
        "metadata": metadata,
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def train_model(
    model,
    train_dataset,
    val_pages: Sequence,
    checkpoint_path: str,
    use_masking: bool,
    cfg: TrainConfig,
    patch_h: int,
    patch_w: int,
    overlap: int = 64,
    metadata: Optional[Dict] = None,
):
    """Train with a fixed crop budget and early-stop on full-page validation F1."""
    if not val_pages:
        raise ValueError("At least one fully annotated validation page is required")

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = make_loader(train_dataset, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    use_scaler = device.type == "cuda" and cfg.mixed_precision and not _cuda_bf16_supported()
    scaler = _make_grad_scaler(use_scaler)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(str(checkpoint) + ".history.json")

    best_f1 = -1.0
    bad_epochs = 0
    history = []
    metadata = dict(metadata or {})

    num_batches = len(train_loader)
    optimizer_updates = int(math.ceil(num_batches / max(1, cfg.grad_accum_steps)))
    print(f"Device: {device}")
    print(f"Samples/epoch: {len(train_dataset)}")
    print(f"Physical batch: {cfg.batch_size}")
    print(f"Gradient accumulation: {max(1, cfg.grad_accum_steps)}")
    print(f"Effective batch (nominal): {cfg.effective_batch_size}")
    print(f"Optimizer updates/epoch: {optimizer_updates}")
    print(f"Early stopping: val_F1, patience={cfg.patience}, min_delta={cfg.min_delta}")

    for epoch in range(1, cfg.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = []
        accum = max(1, cfg.grad_accum_steps)

        for step, (image, target, valid) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            group_start = ((step - 1) // accum) * accum + 1
            group_end = min(group_start + accum - 1, num_batches)
            group_size = group_end - group_start + 1

            with _autocast(device, cfg.mixed_precision):
                logits = model(image)
                loss = partial_cross_entropy(logits, target, valid, use_masking)
                scaled_loss = loss / float(group_size)

            scaler.scale(scaled_loss).backward()
            if step == group_end:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(running)) if running else float("nan")
        val_metrics = evaluate_full_pages(
            model,
            val_pages,
            patch_h,
            patch_w,
            batch_size=cfg.batch_size,
            overlap=overlap,
            mixed_precision=cfg.mixed_precision,
            threshold_step=cfg.threshold_step,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_threshold": val_metrics["threshold"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_F1={val_metrics['f1']:.6f} P={val_metrics['precision']:.6f} "
            f"R={val_metrics['recall']:.6f} th={val_metrics['threshold']:.2f}"
        )

        improved = val_metrics["f1"] > (best_f1 + cfg.min_delta)
        if improved:
            best_f1 = val_metrics["f1"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": getattr(model, "model_name", None),
                    "use_masking": bool(use_masking),
                    "metadata": metadata,
                    "best_epoch": epoch,
                    "best_val_f1": best_f1,
                    "best_val_precision": val_metrics["precision"],
                    "best_val_recall": val_metrics["recall"],
                    "best_threshold": val_metrics["threshold"],
                    "train_config": asdict(cfg),
                },
                checkpoint,
            )
            print(f"  saved best checkpoint: {checkpoint}")
        else:
            bad_epochs += 1
            print(f"  no val_F1 improvement: {bad_epochs}/{cfg.patience}")

        _write_history(history_path, history, cfg, metadata)

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; best val_F1={best_f1:.6f}")
            break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    print(
        f"Restored best checkpoint from epoch {saved['best_epoch']} "
        f"(val_F1={saved['best_val_f1']:.6f}, th={saved['best_threshold']:.2f})"
    )
    return saved
