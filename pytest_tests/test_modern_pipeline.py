import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from modern_fewshot.data import build_legacy_annotation_mask, PartialPage, PartialPatchDataset
from modern_fewshot.inference import best_threshold
from modern_fewshot.losses import partial_cross_entropy
from modern_fewshot.train import TrainConfig


def make_page():
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    # Add some visual variation so crops are not identical.
    image[240:380, 240:380] = 180
    gt = np.zeros((512, 512), dtype=np.uint8)
    gt[260:340, 260:340] = 1
    valid, windows = build_legacy_annotation_mask(gt, 256, 256, 1, 0.001)
    return PartialPage("src", "gt", image, gt, valid, gt * valid, windows)


def test_annotation_mask_selects_requested_windows():
    gt = np.zeros((512, 512), dtype=np.uint8)
    gt[20:120, 20:120] = 1
    gt[300:400, 300:400] = 1
    valid, windows = build_legacy_annotation_mask(gt, 256, 256, 1, 0.001)
    assert len(windows) == 1
    assert valid.sum() == 256 * 256


def test_masked_loss_ignores_unknown_pixels():
    logits = torch.tensor([[[[3.0, -3.0]], [[-3.0, 3.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 0]]])
    valid = torch.tensor([[[1, 0]]], dtype=torch.bool)
    masked = partial_cross_entropy(logits, target, valid, True)
    unmasked = partial_cross_entropy(logits, target, valid, False)
    assert masked.item() < unmasked.item()


def test_all_ablation_samplers_keep_fixed_sample_budget():
    page = make_page()
    uniform = PartialPatchDataset([page], 256, 256, 1024, False, 0.001, ("none",), seed=3)
    guided = PartialPatchDataset([page], 256, 256, 1024, True, 0.001, ("none",), seed=3)
    assert len(uniform) == 1024
    assert len(guided) == 1024


def test_guided_oversampling_returns_annotated_ink():
    page = make_page()
    ds = PartialPatchDataset([page], 256, 256, 32, True, 0.001, ("none",), seed=3)
    for i in range(len(ds)):
        _, target, valid = ds[i]
        assert int(target.sum()) > 0
        assert int(valid.sum()) > 0


def test_sampling_is_reproducible_and_changes_by_epoch():
    page = make_page()
    ds1 = PartialPatchDataset([page], 256, 256, 32, True, 0.001, ("none",), seed=7)
    ds2 = PartialPatchDataset([page], 256, 256, 32, True, 0.001, ("none",), seed=7)
    locs1 = [ds1.sample_location(i) for i in range(8)]
    locs2 = [ds2.sample_location(i) for i in range(8)]
    assert locs1 == locs2
    ds1.set_epoch(1)
    locs_next = [ds1.sample_location(i) for i in range(8)]
    assert locs_next != locs1


def test_validation_threshold_uses_full_gt_when_no_mask_is_passed():
    gt = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    prob = np.array([[0.1, 0.2, 0.8, 0.9]], dtype=np.float32)
    f1, threshold, precision, recall = best_threshold([prob], [gt], valid_masks=None, step=0.1)
    assert f1 > 0.999
    assert precision > 0.999
    assert recall > 0.999
    assert 0.2 < threshold <= 0.8


def test_pretrained_training_defaults_are_conservative():
    cfg = TrainConfig()
    assert cfg.epochs == 50
    assert cfg.patience == 10
    assert cfg.grad_accum_steps == 8
    assert cfg.effective_batch_size == 32


def test_legacy_files_match_recorded_hashes():
    root = Path(__file__).resolve().parents[1]
    for line in (root / "LEGACY_SHA256.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        data = (root / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, relative


def test_legacy_result_header_and_row_keep_old_layout(tmp_path):
    from modern_fewshot.reporting import LEGACY_RESULT_HEADER, legacy_result_line

    assert LEGACY_RESULT_HEADER == (
        "Train;Test;PAG;Num pages train;ANN;Num annotations per page;"
        "PAT;Num random patches;Ink rate;VAL;Th_bin;F1-val;P-val;R-val;"
        "Num annotated patches-val;Maximum num annotated patches-val;TEST;"
        "F1-test;P-test;R-test;Num annotated patches-test;"
        "Maximum num annotated patches-test;IoU-test;Specificity-test;"
        "TP-test;TN-test;FP-test;FN-test;TimePerPage(s);"
    )
    metrics = {
        "f1": 0.8,
        "precision": 0.75,
        "recall": 0.86,
        "iou": 0.67,
        "specificity": 0.95,
        "tp": 10,
        "tn": 20,
        "fp": 3,
        "fn": 2,
    }
    row = legacy_result_line(
        db_train_src="datasets/Dibco/train/SRC",
        db_test_src="datasets/Dibco/test/SRC",
        pages_train=1,
        n_annotated_patches=1,
        npatches=1024,
        ink_rate=0.02,
        threshold=0.5,
        val_f1=0.7,
        val_precision=0.8,
        val_recall=0.6,
        n_annotated_val=1,
        max_annotated_val=8,
        test_metrics=metrics,
        n_annotated_train=1,
        max_annotated_train=9,
        avg_elapsed=1.25,
    )
    assert row.startswith("datasets/Dibco/train/SRC;datasets/Dibco/test/SRC;PAG;1;ANN;1;PAT;1024;0,02;;VAL;")
    assert ";TEST;80,0;75,0;86,0;1;9;67,0;95,0;1000,0;2000,0;300,0;200,0;1,25;" in row


def test_modern_result_images_use_tests_not_pytest_tests(tmp_path, monkeypatch):
    from modern_fewshot.reporting import save_legacy_page_images

    monkeypatch.chdir(tmp_path)
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    gt = np.zeros((8, 8), dtype=np.uint8)
    valid = np.ones((8, 8), dtype=np.uint8)
    prob = np.zeros((8, 8), dtype=np.float32)
    outputs = save_legacy_page_images(
        checkpoint="models/modern/segformer_b0/full/Dibco__seed1.pt",
        kind="tests",
        model="segformer_b0",
        experiment="full",
        dataset_name="Dibco",
        split="test",
        src_root="datasets/Dibco/test/SRC",
        src_path="datasets/Dibco/test/SRC/page.png",
        image_rgb=image,
        gt=gt,
        region_mask=valid,
        probability=prob,
        threshold=0.5,
    )
    assert all(path.startswith("tests/modern/") for path in outputs.values())
    assert all("pytest_tests" not in path for path in outputs.values())
    assert outputs["gt"].endswith("page.png_gt.png")
    assert outputs["gr"].endswith("page.png_gr.png")
    assert outputs["pred"].endswith("page.png_pred.png")
    assert outputs["pred_th"].endswith("page.png_pred_th.png")
    assert outputs["annotated_regions"].endswith("page.png_annotated_regions.png")


def test_model_registry_contains_transformers_and_deeplabv3():
    from modern_fewshot.models import MODEL_REGISTRY

    expected = {
        "segformer_b0",
        "segformer_b1",
        "segformer_b2",
        "segformer_doc_b3",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
    }
    assert expected.issubset(MODEL_REGISTRY)
