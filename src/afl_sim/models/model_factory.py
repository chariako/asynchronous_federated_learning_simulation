from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision
from loguru import logger

from afl_sim.config import ModelConfig
from afl_sim.enums import DatasetType, ModelType


class LogisticRegression(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: int):
        super().__init__()
        self.linear = nn.Linear(image_size * image_size * in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return cast(torch.Tensor, self.linear(x))


class SimpleSequentialCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, image_size: int):
        super().__init__()
        # Using GroupNorm by default
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, image_size, image_size)
            flat_size = self.features(dummy_input).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return cast(torch.Tensor, self.classifier(x))


def _replace_layers(
    model: nn.Module,
    target_types: type[nn.Module] | tuple[type[nn.Module], ...],
    replacement_fn: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Helper function to iterate over a model and replace specific layer types
    using a provided replacement factory function.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, target_types):
            parts = name.split(".")
            child_name = parts[-1]
            parent_name = ".".join(parts[:-1])

            parent = model.get_submodule(parent_name) if parent_name else model

            new_module = replacement_fn(module)
            setattr(parent, child_name, new_module)

    return model


def _remove_norm_layers(model: nn.Module) -> nn.Module:
    """Replaces all BatchNorm and GroupNorm layers with Identity."""
    target_types = (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)
    return _replace_layers(model, target_types, lambda m: nn.Identity())


def _bn_to_gn(model: nn.Module, groups: int = 32) -> nn.Module:
    """Replaces all BatchNorm layers with GroupNorm."""

    def create_group_norm(module: nn.Module) -> nn.GroupNorm:
        num_channels = getattr(module, "num_features", None)
        if num_channels is None:
            raise ValueError(f"Module {module} does not have 'num_features'")

        if num_channels % 32 == 0:
            groups = 32
        elif num_channels < 32:
            groups = 1
        else:
            groups = 1
            for g in [32, 16, 8, 4, 2]:
                if num_channels % g == 0:
                    groups = g
                    break

        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)

    return _replace_layers(model, nn.modules.batchnorm._BatchNorm, create_group_norm)


def _adapt_stem_for_cifar(model: nn.Module, model_type: str) -> None:
    """
    Modifies the input stem of standard ImageNet models to work better
    with small images (32x32) like CIFAR-10.
    """
    model_dynamic = cast(Any, model)

    if model_type == ModelType.RESNET18:
        # Replace the first 7x7 conv with a smaller 3x3 conv
        if hasattr(model_dynamic, "conv1") and isinstance(
            model_dynamic.conv1, nn.Conv2d
        ):
            old_conv = model_dynamic.conv1
            model_dynamic.conv1 = nn.Conv2d(
                old_conv.in_channels,
                old_conv.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        if hasattr(model_dynamic, "maxpool"):
            model_dynamic.maxpool = nn.Identity()

    elif model_type == ModelType.MOBILENET_V2:
        # Reduce stride in the first convolution
        if hasattr(model_dynamic, "features") and len(model_dynamic.features) > 0:
            first_conv = model_dynamic.features[0][0]
            if isinstance(first_conv, nn.Conv2d):
                first_conv.stride = (1, 1)


def get_model(dataset: DatasetType, model_config: ModelConfig) -> nn.Module:
    """
    Factory function to initialize models based on configuration.
    """
    model_type = model_config.model_name
    stress_test = model_config.stress_test

    if stress_test:
        logger.warning(
            f"Stress Test Active: Removing GroupNorm/BatchNorm from {model_type}"
        )

    # --- Logistic Regression ---
    if model_type == ModelType.LOG_REG:
        return LogisticRegression(
            in_channels=dataset.num_channels,
            num_classes=dataset.num_classes,
            image_size=dataset.image_size,
        )

    # --- Simple CNN ---
    if model_type == ModelType.CNN:
        model = SimpleSequentialCNN(
            in_channels=dataset.num_channels,
            num_classes=dataset.num_classes,
            image_size=dataset.image_size,
        )
        if stress_test:
            return _remove_norm_layers(model)
        return model

    # --- ResNet / MobileNet ---
    logger.info(f"Initializing {model_type} for dataset '{dataset}'...")

    if model_type == ModelType.RESNET18:
        model = torchvision.models.resnet18(weights=None)
        # Replace the final fully connected layer
        model.fc = nn.Linear(
            model.fc.in_features, out_features=dataset.num_classes, bias=True
        )

    elif model_type == ModelType.MOBILENET_V2:
        model = torchvision.models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, out_features=dataset.num_classes, bias=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Adapt for small images (CIFAR-10/100) if necessary
    if dataset.image_size <= 64:
        logger.info(
            f"Adapting {model_type} stem for small input size ({dataset.image_size}x{dataset.image_size})"
        )
        _adapt_stem_for_cifar(model, model_type)

    # Handle Normalization Layers
    if stress_test:
        return _remove_norm_layers(model)

    logger.info(f"Replacing BatchNorm with GroupNorm in '{model_type}'...")
    return _bn_to_gn(model)
