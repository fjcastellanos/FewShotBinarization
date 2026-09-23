from __future__ import annotations

from typing import Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


SEGFORMER_REGISTRY: Dict[str, str] = {
    "segformer_b0": "nvidia/mit-b0",
    "segformer_b1": "nvidia/mit-b1",
    "segformer_b2": "nvidia/mit-b2",
    "segformer_doc_b3": "DiTo97/binarization-segformer-b3",
}

DEEPLABV3_REGISTRY: Dict[str, str] = {
    "deeplabv3_resnet50": "torchvision/deeplabv3_resnet50",
    "deeplabv3_resnet101": "torchvision/deeplabv3_resnet101",
}

# Public registry used by argparse and the experiment scripts.
MODEL_REGISTRY: Dict[str, str] = {
    **SEGFORMER_REGISTRY,
    **DEEPLABV3_REGISTRY,
}


class SegFormerBinary(nn.Module):
    def __init__(self, model_name: str, gradient_checkpointing: bool = False):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "SegFormer requires `transformers`. Install requirements-modern.txt"
            ) from exc

        if model_name not in SEGFORMER_REGISTRY:
            raise ValueError(
                f"Unknown SegFormer model {model_name!r}; choices: {sorted(SEGFORMER_REGISTRY)}"
            )
        checkpoint = SEGFORMER_REGISTRY[model_name]
        self.model_name = model_name
        self.checkpoint = checkpoint

        processor = AutoImageProcessor.from_pretrained(checkpoint)
        mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
        std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if model_name == "segformer_doc_b3":
            self.net = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
            if self.net.config.num_labels != 2:
                raise ValueError(f"Expected 2 labels in {checkpoint}, got {self.net.config.num_labels}")
        else:
            config = SegformerConfig.from_pretrained(
                checkpoint,
                num_labels=2,
                id2label={0: "background", 1: "foreground"},
                label2id={"background": 0, "foreground": 1},
            )
            # nvidia/mit-b* contains the ImageNet-pretrained encoder. The
            # segmentation decode head is intentionally initialized for this task.
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                config=config,
                ignore_mismatched_sizes=True,
            )

        if gradient_checkpointing and hasattr(self.net, "gradient_checkpointing_enable"):
            self.net.gradient_checkpointing_enable()

    def freeze_encoder(self, freeze: bool = True):
        for p in self.net.segformer.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        logits = self.net(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class DeepLabV3Binary(nn.Module):
    """Torchvision DeepLabV3 adapted to the same two-class interface as SegFormer.

    Only the ResNet backbone is ImageNet-pretrained. The DeepLabV3 segmentation
    head is initialized for this two-class document task. This mirrors the clean
    SegFormer B0/B1/B2 comparison, where the encoder is pretrained and the task
    head is initialized for binarization.
    """

    def __init__(self, model_name: str, gradient_checkpointing: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, ResNet101_Weights
            from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
        except ImportError as exc:
            raise ImportError(
                "DeepLabV3 requires a torchvision version compatible with PyTorch. "
                "Install requirements-modern.txt"
            ) from exc

        if model_name not in DEEPLABV3_REGISTRY:
            raise ValueError(
                f"Unknown DeepLabV3 model {model_name!r}; choices: {sorted(DEEPLABV3_REGISTRY)}"
            )

        self.model_name = model_name
        self.checkpoint = DEEPLABV3_REGISTRY[model_name]

        # Torchvision's pretrained ResNet backbones use ImageNet normalization.
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if model_name == "deeplabv3_resnet50":
            self.net = deeplabv3_resnet50(
                weights=None,
                weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
                num_classes=2,
                aux_loss=False,
            )
        else:
            self.net = deeplabv3_resnet101(
                weights=None,
                weights_backbone=ResNet101_Weights.IMAGENET1K_V1,
                num_classes=2,
                aux_loss=False,
            )

        if gradient_checkpointing:
            warnings.warn(
                "--gradient-checkpointing is not supported by torchvision DeepLabV3 "
                "and is ignored for this model.",
                RuntimeWarning,
                stacklevel=2,
            )

    def freeze_encoder(self, freeze: bool = True):
        for p in self.net.backbone.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = (x - self.image_mean.to(dtype=x.dtype)) / self.image_std.to(dtype=x.dtype)
        output = self.net(x)
        logits = output["out"] if isinstance(output, dict) else output
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def create_model(
    model_name: str,
    freeze_encoder: bool = False,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    if model_name in SEGFORMER_REGISTRY:
        model: nn.Module = SegFormerBinary(
            model_name,
            gradient_checkpointing=gradient_checkpointing,
        )
    elif model_name in DEEPLABV3_REGISTRY:
        model = DeepLabV3Binary(
            model_name,
            gradient_checkpointing=gradient_checkpointing,
        )
    else:
        raise ValueError(f"Unknown model {model_name!r}; choices: {sorted(MODEL_REGISTRY)}")

    model.freeze_encoder(freeze_encoder)
    return model
