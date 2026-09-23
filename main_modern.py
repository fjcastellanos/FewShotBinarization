#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from modern_fewshot.data import (
    PartialPage,
    PartialPatchDataset,
    build_legacy_annotation_mask,
    list_files_recursive,
    load_binary_gt,
    load_rgb,
    match_src_gt,
    order_pairs_by_resolution,
)
from modern_fewshot.inference import predict_page_tiled
from modern_fewshot.models import MODEL_REGISTRY, create_model
from modern_fewshot.reporting import (
    LEGACY_RESULT_HEADER,
    append_line,
    binary_metrics,
    count_annotation_windows,
    infer_dataset_name,
    legacy_result_line,
    sanitize_token,
    save_legacy_page_images,
)
from modern_fewshot.train import TrainConfig, train_model


EXPERIMENTS = {
    "baseline": (False, False),       # no masking, no oversampling
    "oversampling": (False, True),    # oversampling only
    "full": (True, True),             # masking + oversampling
    "masking": (True, False),         # fourth cell of the 2x2 ablation
}


def parser():
    p = argparse.ArgumentParser(
        description=(
            "Modern few-shot document binarization. The historical Keras/SAE "
            "entry point main.py is preserved unchanged."
        )
    )
    # Familiar legacy-style names where practical.
    p.add_argument("-db_train_src", "--db-train-src", required=True)
    p.add_argument("-db_train_gt", "--db-train-gt", required=True)
    p.add_argument("-db_test_src", "--db-test-src")
    p.add_argument("-db_test_gt", "--db-test-gt")
    p.add_argument("-pages_train", "--pages-train", type=int, default=1)
    p.add_argument("--val-pages", type=int, default=1)
    p.add_argument(
        "-npatches", "--npatches", type=int, default=1024,
        help="Virtual samples per training page and epoch; same budget for all ablations",
    )
    p.add_argument("-n_annotated_patches", "--n-annotated-patches", type=int, default=1)
    p.add_argument("-window_w", "--window-w", type=int, default=256)
    p.add_argument("-window_h", "--window-h", type=int, default=256)
    p.add_argument("-ink_rate", "--ink-rate", type=float, default=0.02)
    p.add_argument(
        "-aug", "--aug", nargs="*", default=["random"],
        choices=["none", "random", "flipH", "flipV", "rot", "scale"],
        help=(
            "Geometric augmentation choices. 'random' is kept as a legacy compatibility token; "
            "oversampling itself is controlled by --experiment."
        ),
    )

    p.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="segformer_b0")
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="full")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("-e", "--epochs", type=int, default=50,
                   help="Maximum epochs. Early stopping normally stops earlier.")
    p.add_argument("-b", "--batch-size", type=int, default=4,
                   help="Physical batch size (GPU-memory dependent)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation; default 4x8 gives nominal effective batch 32")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="Minimum full-page validation F1 increase that resets patience")
    p.add_argument("--threshold-step", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overlap", type=int, default=64,
                   help="Tile overlap for full-page validation/test inference")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset-name", default=None,
                   help="Name used to namespace checkpoints/results; inferred from .../<dataset>/train/SRC by default")
    p.add_argument("-res", "--res", default=None,
                   help="Append one legacy-format semicolon-separated result row to this file")
    p.add_argument("--no-save-result-images", action="store_true",
                   help="Do not save legacy-style test PNGs under tests/modern/")
    p.add_argument("--no-save-train-images", action="store_true",
                   help="Do not save GT/source/annotation visualizations under train/modern/")
    p.add_argument("--test", action="store_true", help="Do not train; load checkpoint and evaluate")
    return p


def build_partial_pages(pairs, args):
    return [
        PartialPage.from_files(
            src,
            gt,
            patch_h=args.window_h,
            patch_w=args.window_w,
            n_annotated_patches=args.n_annotated_patches,
            min_ink_rate=args.ink_rate,
        )
        for src, gt in pairs
    ]


def build_validation_pages(pairs, args):
    # Early stopping uses complete GT. Partial fields are deliberately not used.
    return [
        PartialPage.from_files(
            src,
            gt,
            patch_h=args.window_h,
            patch_w=args.window_w,
            n_annotated_patches=-1,
            min_ink_rate=0.0,
        )
        for src, gt in pairs
    ]


def default_checkpoint(args, dataset_name: str):
    aug = "-".join(args.aug) if args.aug else "none"
    aug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in aug)
    ink = str(args.ink_rate).replace(".", "p")
    filename = (
        f"{sanitize_token(dataset_name)}__pt{args.pages_train}"
        f"__nap{args.n_annotated_patches}__np{args.npatches}"
        f"__win{args.window_w}x{args.window_h}__ink{ink}"
        f"__aug-{aug}__freeze{int(args.freeze_encoder)}__seed{args.seed}.pt"
    )
    return str(Path("models") / "modern" / args.model / args.experiment / filename)


def save_train_and_validation_visualizations(
    train_pages,
    val_pairs,
    checkpoint,
    dataset_name,
    args,
):
    if args.no_save_train_images:
        return

    for page in train_pages:
        save_legacy_page_images(
            checkpoint=checkpoint,
            kind="train",
            model=args.model,
            experiment=args.experiment,
            dataset_name=dataset_name,
            split="train",
            src_root=args.db_train_src,
            src_path=page.src_path,
            image_rgb=page.image,
            gt=page.full_gt,
            region_mask=page.valid_mask,
        )

    # Legacy threshold computation also writes the validation page GT/source and
    # simulated annotation mask into train/. Keep that convention here.
    for src, gt_path in val_pairs:
        image = load_rgb(src)
        gt = load_binary_gt(gt_path)
        region_mask, _ = build_legacy_annotation_mask(
            gt,
            args.window_h,
            args.window_w,
            args.n_annotated_patches,
            args.ink_rate,
        )
        save_legacy_page_images(
            checkpoint=checkpoint,
            kind="train",
            model=args.model,
            experiment=args.experiment,
            dataset_name=dataset_name,
            split="train",
            src_root=args.db_train_src,
            src_path=src,
            image_rgb=image,
            gt=gt,
            region_mask=region_mask,
        )


def evaluate_pages(model, pages, threshold, args):
    all_pred = []
    all_gt = []
    for page in pages:
        prob = predict_page_tiled(
            model,
            page.image,
            args.window_h,
            args.window_w,
            batch_size=args.batch_size,
            overlap=args.overlap,
            mixed_precision=not args.no_amp,
        )
        # Historical util.run_test uses a strict comparison.
        all_pred.append((prob > threshold).reshape(-1))
        all_gt.append(page.full_gt.reshape(-1).astype(bool))
    if not all_pred:
        raise ValueError("No pages available for evaluation")
    return binary_metrics(np.concatenate(all_pred), np.concatenate(all_gt))


def evaluate_test_set(model, pairs, threshold, checkpoint, dataset_name, args):
    all_pred = []
    all_gt = []
    per_page = []
    elapsed_times = []

    for src, gt_path in pairs:
        image = load_rgb(src)
        gt = load_binary_gt(gt_path)
        start = perf_counter()
        prob = predict_page_tiled(
            model,
            image,
            args.window_h,
            args.window_w,
            batch_size=args.batch_size,
            overlap=args.overlap,
            mixed_precision=not args.no_amp,
        )
        elapsed = perf_counter() - start
        elapsed_times.append(elapsed)

        pred = prob > threshold
        metrics = binary_metrics(pred, gt)
        page_record = {
            "page": Path(src).name,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "iou": metrics["iou"],
            "specificity": metrics["specificity"],
            "tp": metrics["tp"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "elapsed_seconds": elapsed,
        }

        if not args.no_save_result_images:
            region_mask, _ = build_legacy_annotation_mask(
                gt,
                args.window_h,
                args.window_w,
                args.n_annotated_patches,
                args.ink_rate,
            )
            page_record["images"] = save_legacy_page_images(
                checkpoint=checkpoint,
                kind="tests",
                model=args.model,
                experiment=args.experiment,
                dataset_name=dataset_name,
                split="test",
                src_root=args.db_test_src,
                src_path=src,
                image_rgb=image,
                gt=gt,
                region_mask=region_mask,
                probability=prob,
                threshold=threshold,
            )

        per_page.append(page_record)
        all_pred.append(pred.reshape(-1))
        all_gt.append(gt.reshape(-1).astype(bool))
        print(
            f"TEST page={Path(src).name} F1={metrics['f1']:.6f} "
            f"P={metrics['precision']:.6f} R={metrics['recall']:.6f} "
            f"IoU={metrics['iou']:.6f} time={elapsed:.6f}s"
        )

    global_metrics = binary_metrics(np.concatenate(all_pred), np.concatenate(all_gt))
    global_metrics["per_page"] = per_page
    global_metrics["avg_elapsed_seconds"] = float(np.mean(elapsed_times)) if elapsed_times else 0.0
    return global_metrics


def main():
    args = parser().parse_args()
    use_masking, use_oversampling = EXPERIMENTS[args.experiment]
    dataset_name = sanitize_token(args.dataset_name or infer_dataset_name(args.db_train_src))
    checkpoint = args.checkpoint or default_checkpoint(args, dataset_name)

    if args.pages_train < 1:
        raise ValueError("--pages-train must be >= 1; the normal experiment uses 1")
    if args.val_pages < 1:
        raise ValueError("--val-pages must be >= 1")
    if args.npatches <= 0:
        raise ValueError("--npatches must be > 0 in the fair fixed-budget modern ablation")
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("--batch-size and --grad-accum must be > 0")

    train_pairs = match_src_gt(
        list_files_recursive(args.db_train_src),
        list_files_recursive(args.db_train_gt),
    )
    train_pairs = order_pairs_by_resolution(train_pairs)
    required = args.pages_train + args.val_pages
    if len(train_pairs) < required:
        raise ValueError(f"Need at least {required} train/validation pages, found {len(train_pairs)}")

    selected_train = train_pairs[:args.pages_train]
    selected_val = train_pairs[args.pages_train:required]
    train_pages = build_partial_pages(selected_train, args)
    val_pages = build_validation_pages(selected_val, args)
    samples_per_epoch = args.npatches * len(train_pages)

    n_annotated_train = sum(len(page.annotated_windows) for page in train_pages)
    max_annotated_train = count_annotation_windows(
        selected_train, load_binary_gt, args.window_h, args.window_w, -1, args.ink_rate
    )
    n_annotated_val = count_annotation_windows(
        selected_val, load_binary_gt, args.window_h, args.window_w,
        args.n_annotated_patches, args.ink_rate,
    )
    max_annotated_val = count_annotation_windows(
        selected_val, load_binary_gt, args.window_h, args.window_w, -1, args.ink_rate
    )

    print("CONFIG")
    print(f"  dataset={dataset_name}")
    print(f"  model={args.model}")
    print(f"  experiment={args.experiment} masking={use_masking} oversampling={use_oversampling}")
    print(f"  checkpoint={checkpoint}")
    print(f"  train_pages={len(train_pages)} val_pages={len(val_pages)}")
    print(f"  virtual_samples_per_page={args.npatches}")
    print(f"  total_samples_per_epoch={samples_per_epoch}")
    print(
        f"  physical_batch={args.batch_size} grad_accum={args.grad_accum} "
        f"effective_batch_nominal={args.batch_size * args.grad_accum}"
    )
    print(f"  max_epochs={args.epochs} patience={args.patience} min_delta={args.min_delta}")
    print("  early_stopping_monitor=full_page_val_F1")
    for i, page in enumerate(train_pages):
        ratio = float(page.valid_mask.mean())
        print(
            f"  train_page[{i}]={Path(page.src_path).name} "
            f"annotated_windows={len(page.annotated_windows)} annotated_ratio={ratio:.6f}"
        )

    model = create_model(
        args.model,
        freeze_encoder=args.freeze_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if not args.test:
        train_ds = PartialPatchDataset(
            train_pages,
            args.window_h,
            args.window_w,
            samples_per_epoch=samples_per_epoch,
            oversampling=use_oversampling,
            min_ink_rate=args.ink_rate,
            augmentations=args.aug,
            seed=args.seed,
        )
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            grad_accum_steps=args.grad_accum,
            mixed_precision=not args.no_amp,
            num_workers=args.num_workers,
            seed=args.seed,
            threshold_step=args.threshold_step,
        )
        metadata = vars(args).copy()
        metadata.update(
            {
                "dataset_name": dataset_name,
                "use_masking": use_masking,
                "use_oversampling": use_oversampling,
                "samples_per_epoch": samples_per_epoch,
                "effective_batch_nominal": args.batch_size * args.grad_accum,
                "validation_scope": "full_page_gt",
                "early_stopping_metric": "val_f1",
            }
        )
        train_model(
            model,
            train_ds,
            val_pages,
            checkpoint,
            use_masking,
            cfg,
            patch_h=args.window_h,
            patch_w=args.window_w,
            overlap=args.overlap,
            metadata=metadata,
        )

    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    threshold = float(saved.get("best_threshold", 0.5))
    print(
        f"BEST checkpoint epoch={saved.get('best_epoch', '?')} "
        f"val_F1={saved.get('best_val_f1', float('nan')):.6f} threshold={threshold:.2f}"
    )

    # Re-evaluate the selected validation pages with legacy metric conventions
    # so the exported row is directly comparable to previous result files.
    legacy_val_metrics = evaluate_pages(model, val_pages, threshold, args)
    save_train_and_validation_visualizations(
        train_pages, selected_val, checkpoint, dataset_name, args
    )

    summary = {
        "checkpoint": checkpoint,
        "dataset": dataset_name,
        "model": args.model,
        "experiment": args.experiment,
        "threshold": threshold,
        "best_epoch": saved.get("best_epoch"),
        "best_val_f1_selection": saved.get("best_val_f1"),
        "best_val_precision_selection": saved.get("best_val_precision"),
        "best_val_recall_selection": saved.get("best_val_recall"),
        "legacy_format_val": legacy_val_metrics,
        "samples_per_epoch": samples_per_epoch,
        "effective_batch_nominal": args.batch_size * args.grad_accum,
        "n_annotated_train": n_annotated_train,
        "max_annotated_train": max_annotated_train,
        "n_annotated_val": n_annotated_val,
        "max_annotated_val": max_annotated_val,
    }

    if args.db_test_src and args.db_test_gt:
        test_pairs = match_src_gt(
            list_files_recursive(args.db_test_src),
            list_files_recursive(args.db_test_gt),
        )
        test_metrics = evaluate_test_set(
            model, test_pairs, threshold, checkpoint, dataset_name, args
        )
        summary["test"] = test_metrics
        print(
            f"TEST GLOBAL F1={test_metrics['f1']:.6f} "
            f"P={test_metrics['precision']:.6f} R={test_metrics['recall']:.6f} "
            f"IoU={test_metrics['iou']:.6f} Specificity={test_metrics['specificity']:.6f}"
        )
        print(f"Duration AVG: {test_metrics['avg_elapsed_seconds']:.6f} s")

        result_line = legacy_result_line(
            db_train_src=args.db_train_src,
            db_test_src=args.db_test_src,
            pages_train=args.pages_train,
            n_annotated_patches=args.n_annotated_patches,
            npatches=args.npatches,
            ink_rate=args.ink_rate,
            threshold=threshold,
            val_f1=legacy_val_metrics["f1"],
            val_precision=legacy_val_metrics["precision"],
            val_recall=legacy_val_metrics["recall"],
            n_annotated_val=n_annotated_val,
            max_annotated_val=max_annotated_val,
            test_metrics=test_metrics,
            n_annotated_train=n_annotated_train,
            max_annotated_train=max_annotated_train,
            avg_elapsed=test_metrics["avg_elapsed_seconds"],
        )
        print("*" * 80)
        print(LEGACY_RESULT_HEADER)
        print(result_line)
        summary["legacy_result_header"] = LEGACY_RESULT_HEADER
        summary["legacy_result_line"] = result_line
        if args.res:
            append_line(args.res, result_line)
            summary["legacy_result_file"] = args.res
            print(f"Legacy-format result appended to: {args.res}")

    summary_path = Path(str(checkpoint) + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
