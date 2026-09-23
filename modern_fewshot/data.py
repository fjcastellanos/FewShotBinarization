from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


Window = Tuple[int, int, int, int]  # top, left, height, width
SampleLocation = Tuple[int, int, int]  # page_idx, top, left


def list_files_recursive(path: str) -> List[str]:
    files = [str(p) for p in Path(path).rglob("*") if p.is_file()]
    files.sort()
    return files


def match_src_gt(src_files: Sequence[str], gt_files: Sequence[str]) -> List[Tuple[str, str]]:
    if len(src_files) != len(gt_files):
        raise ValueError(f"Different number of SRC ({len(src_files)}) and GT ({len(gt_files)}) files")
    pairs: List[Tuple[str, str]] = []
    for src, gt in zip(src_files, gt_files):
        if Path(src).name != Path(gt).name:
            raise ValueError(f"SRC/GT basename mismatch: {src} vs {gt}")
        pairs.append((src, gt))
    return pairs


def order_pairs_by_resolution(pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Match the legacy behavior: larger pages first."""
    scored = []
    for pair in pairs:
        image = cv2.imread(pair[0], cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot read image: {pair[0]}")
        scored.append((pair, int(image.shape[0] * image.shape[1])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored]


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_binary_gt(path: str) -> np.ndarray:
    """Load foreground=1, background=0 using the legacy <128 convention."""
    gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise OSError(f"Cannot read ground truth: {path}")
    if gt.ndim == 3:
        # The old source comments mention alpha-channel annotations. Prefer
        # alpha when present; otherwise convert the color GT to gray.
        if gt.shape[2] == 4:
            gt = gt[:, :, 3]
        else:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    return (gt < 128).astype(np.uint8)


def build_legacy_annotation_mask(
    gt: np.ndarray,
    patch_h: int,
    patch_w: int,
    n_annotated_patches: int,
    min_ink_rate: float,
) -> Tuple[np.ndarray, List[Window]]:
    """Reproduce the legacy sequential selection of manually annotated windows.

    The full GT is used only to *simulate* which rectangular regions have been
    manually annotated. Only selected rectangles become valid supervision.
    """
    rows, cols = gt.shape[:2]
    valid = np.zeros((rows, cols), dtype=np.uint8)
    selected: List[Window] = []

    for center_r in range(patch_h // 2, rows + patch_h // 2 - 1, patch_h):
        for center_c in range(patch_w // 2, cols + patch_w // 2 - 1, patch_w):
            r = min(center_r, rows - patch_h // 2)
            c = min(center_c, cols - patch_w // 2)
            top = max(0, r - patch_h // 2)
            left = max(0, c - patch_w // 2)
            bottom = min(rows, top + patch_h)
            right = min(cols, left + patch_w)
            top = max(0, bottom - patch_h)
            left = max(0, right - patch_w)

            sample = gt[top:bottom, left:right]
            if sample.size == 0 or np.count_nonzero(sample == 1) == 0:
                continue
            ink_rate = float(np.count_nonzero(sample == 1)) / float(patch_h * patch_w)
            if ink_rate < min_ink_rate:
                continue

            valid[top:bottom, left:right] = 1
            selected.append((top, left, bottom - top, right - left))
            if n_annotated_patches not in (-1, 0) and len(selected) >= n_annotated_patches:
                return valid, selected

    return valid, selected


def crop_with_padding(arr: np.ndarray, top: int, left: int, h: int, w: int, fill: int = 0) -> np.ndarray:
    """Crop an array and pad to exactly (h,w) if it touches an image border."""
    out_shape = (h, w) + (() if arr.ndim == 2 else (arr.shape[2],))
    out = np.full(out_shape, fill, dtype=arr.dtype)
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(arr.shape[0], top + h)
    src_right = min(arr.shape[1], left + w)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    dst_top = src_top - top
    dst_left = src_left - left
    out[
        dst_top:dst_top + (src_bottom - src_top),
        dst_left:dst_left + (src_right - src_left),
    ] = arr[src_top:src_bottom, src_left:src_right]
    return out


def _center_crop_or_pad(arr: np.ndarray, h: int, w: int, fill: int) -> np.ndarray:
    top = (arr.shape[0] - h) // 2
    left = (arr.shape[1] - w) // 2
    return crop_with_padding(arr, top, left, h, w, fill=fill)


def _rotate_bound(arr: np.ndarray, angle: float, interpolation: int, border_value) -> np.ndarray:
    """Legacy-style bound-preserving rotation, then caller crops back to patch size."""
    h, w = arr.shape[:2]
    c_x, c_y = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - c_x
    matrix[1, 2] += (new_h / 2.0) - c_y
    return cv2.warpAffine(
        arr,
        matrix,
        (new_w, new_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


@dataclass
class PartialPage:
    src_path: str
    gt_path: str
    image: np.ndarray
    full_gt: np.ndarray
    valid_mask: np.ndarray
    partial_gt: np.ndarray
    annotated_windows: List[Window]

    @classmethod
    def from_files(
        cls,
        src_path: str,
        gt_path: str,
        patch_h: int,
        patch_w: int,
        n_annotated_patches: int,
        min_ink_rate: float,
    ) -> "PartialPage":
        image = load_rgb(src_path)
        full_gt = load_binary_gt(gt_path)
        if image.shape[:2] != full_gt.shape[:2]:
            raise ValueError(
                f"SRC/GT size mismatch: {src_path} {image.shape[:2]} vs {gt_path} {full_gt.shape[:2]}"
            )
        valid_mask, windows = build_legacy_annotation_mask(
            full_gt, patch_h, patch_w, n_annotated_patches, min_ink_rate
        )
        partial_gt = (full_gt * valid_mask).astype(np.uint8)
        return cls(src_path, gt_path, image, full_gt, valid_mask, partial_gt, windows)


class PatchAugmenter:
    """Geometric augmentations with the same choices/ranges as the legacy code.

    `random` is accepted only as a legacy compatibility token. In the old code
    it selected annotation-guided random sampling; in the modern pipeline the
    sampling strategy is controlled explicitly by --experiment.
    """

    def __init__(self, modes: Sequence[str]):
        self.modes = tuple(modes)

    def _apply_geom(self, image, target, valid, mode, rng: random.Random):
        h, w = target.shape
        if mode == "flipH":
            return (
                np.ascontiguousarray(image[:, ::-1]),
                np.ascontiguousarray(target[:, ::-1]),
                np.ascontiguousarray(valid[:, ::-1]),
            )
        if mode == "flipV":
            # Preserve the historical implementation: cv2.flip(..., -1), i.e.
            # both axes (180-degree flip), despite the legacy name "flipV".
            return (
                np.ascontiguousarray(image[::-1, ::-1]),
                np.ascontiguousarray(target[::-1, ::-1]),
                np.ascontiguousarray(valid[::-1, ::-1]),
            )
        if mode == "rot":
            angle = rng.uniform(-5.0, 5.0)
            image_r = _rotate_bound(image, angle, cv2.INTER_LINEAR, (255, 255, 255))
            target_r = _rotate_bound(target, angle, cv2.INTER_NEAREST, 0)
            valid_r = _rotate_bound(valid, angle, cv2.INTER_NEAREST, 0)
            return (
                _center_crop_or_pad(image_r, h, w, 255),
                _center_crop_or_pad(target_r, h, w, 0),
                _center_crop_or_pad(valid_r, h, w, 0),
            )
        if mode == "scale":
            factor = rng.uniform(0.90, 1.10)
            new_w = max(1, int(round(w * factor)))
            new_h = max(1, int(round(h * factor)))
            image_s = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target_s = cv2.resize(target, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            valid_s = cv2.resize(valid, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return (
                _center_crop_or_pad(image_s, h, w, 255),
                _center_crop_or_pad(target_s, h, w, 0),
                _center_crop_or_pad(valid_s, h, w, 0),
            )
        return image, target, valid

    def __call__(self, image, target, valid, rng: random.Random):
        modes = [m for m in self.modes if m not in ("none", "random")]
        rng.shuffle(modes)
        for mode in modes:
            if rng.randint(0, 1) == 1:
                image, target, valid = self._apply_geom(image, target, valid, mode, rng)
        return image, target, valid


class PartialPatchDataset(Dataset):
    """Virtual fixed-budget patch dataset for the 2x2 ablation.

    Every experiment has exactly ``samples_per_epoch`` items. With
    ``oversampling=False`` patches are sampled uniformly from the training page(s).
    With ``oversampling=True`` they are sampled around annotated foreground, using
    the same row/column offsets and ink-rate acceptance idea as the legacy SAE.

    Masking is deliberately *not* implemented here. The dataset always returns an
    explicit valid_mask and the training loss decides whether to use it.
    """

    def __init__(
        self,
        pages: Sequence[PartialPage],
        patch_h: int,
        patch_w: int,
        samples_per_epoch: int,
        oversampling: bool,
        min_ink_rate: float,
        augmentations: Sequence[str] = ("none",),
        seed: int = 1,
        legacy_row_offset: int = 100,
        legacy_col_offset: int = 50,
        max_tries: int = 100,
    ):
        if not pages:
            raise ValueError("At least one page is required")
        self.pages = list(pages)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.oversampling = bool(oversampling)
        self.min_ink_rate = float(min_ink_rate)
        self.seed = int(seed)
        self.samples_per_epoch = int(samples_per_epoch)
        self.legacy_row_offset = int(legacy_row_offset)
        self.legacy_col_offset = int(legacy_col_offset)
        self.max_tries = int(max_tries)
        self.augmenter = PatchAugmenter(augmentations)
        self.epoch = 0

        self._annotated: List[Tuple[int, Window]] = []
        for page_idx, page in enumerate(self.pages):
            self._annotated.extend((page_idx, window) for window in page.annotated_windows)
        if not self._annotated:
            raise ValueError("No annotated patch satisfies the requested ink rate")

        if self.samples_per_epoch <= 0:
            self.samples_per_epoch = len(self._annotated)

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch: int):
        """Make virtual samples change by epoch while remaining reproducible."""
        self.epoch = int(epoch)

    def _rng(self, index: int) -> random.Random:
        value = (
            self.seed * 1_000_003
            + self.epoch * 97_409
            + int(index) * 9_176
            + 17
        ) & 0xFFFFFFFFFFFFFFFF
        return random.Random(value)

    def _uniform_crop(self, rng: random.Random) -> SampleLocation:
        page_idx = rng.randrange(len(self.pages))
        page = self.pages[page_idx]
        max_top = max(0, page.image.shape[0] - self.patch_h)
        max_left = max(0, page.image.shape[1] - self.patch_w)
        top = rng.randint(0, max_top) if max_top > 0 else 0
        left = rng.randint(0, max_left) if max_left > 0 else 0
        return page_idx, top, left

    def _oversampled_crop(self, rng: random.Random) -> SampleLocation:
        eligible_indices = [
            i for i, page in enumerate(self.pages)
            if np.count_nonzero(page.partial_gt == 1) > 0
        ]
        if not eligible_indices:
            return self._uniform_crop(rng)

        page_idx = eligible_indices[rng.randrange(len(eligible_indices))]
        page = self.pages[page_idx]
        ys, xs = np.where(page.partial_gt == 1)
        area = float(self.patch_h * self.patch_w)
        top = left = 0

        for _ in range(self.max_tries + 1):
            j = rng.randrange(len(ys))
            center_r = int(ys[j]) - self.legacy_row_offset
            center_c = int(xs[j]) - self.legacy_col_offset

            if page.image.shape[0] > self.patch_h:
                center_r = max(self.patch_h // 2 + 1, center_r)
                center_r = min(page.image.shape[0] - self.patch_h // 2 - 1, center_r)
                top = center_r - self.patch_h // 2
            else:
                top = 0

            if page.image.shape[1] > self.patch_w:
                center_c = max(self.patch_w // 2 + 1, center_c)
                center_c = min(page.image.shape[1] - self.patch_w // 2 - 1, center_c)
                left = center_c - self.patch_w // 2
            else:
                left = 0

            target = crop_with_padding(
                page.partial_gt, top, left, self.patch_h, self.patch_w, fill=0
            )
            if float(np.count_nonzero(target == 1)) / area >= self.min_ink_rate:
                return page_idx, top, left

        # Legacy code accepts the last candidate after MAX_TRIES.
        return page_idx, top, left

    def sample_location(self, index: int) -> SampleLocation:
        """Expose coordinates for reproducibility/tests without loading tensors."""
        rng = self._rng(index)
        return self._oversampled_crop(rng) if self.oversampling else self._uniform_crop(rng)

    def __getitem__(self, index):
        rng = self._rng(index)
        if self.oversampling:
            page_idx, top, left = self._oversampled_crop(rng)
        else:
            page_idx, top, left = self._uniform_crop(rng)
        page = self.pages[page_idx]

        image = crop_with_padding(page.image, top, left, self.patch_h, self.patch_w, fill=255)
        # Unknown/unannotated pixels deliberately contain target=0. If masking is
        # disabled they therefore enter the loss as background: this is exactly
        # the no-masking ablation. valid_mask remains the source of truth.
        target = crop_with_padding(page.partial_gt, top, left, self.patch_h, self.patch_w, fill=0)
        valid = crop_with_padding(page.valid_mask, top, left, self.patch_h, self.patch_w, fill=0)
        image, target, valid = self.augmenter(image, target, valid, rng)

        # CHW float in [0,1]. Model-specific normalization is applied by the
        # model adapter, keeping data/model responsibilities separate.
        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        target_t = torch.from_numpy(np.ascontiguousarray(target)).long()
        valid_t = torch.from_numpy(np.ascontiguousarray(valid)).bool()
        return image_t, target_t, valid_t
