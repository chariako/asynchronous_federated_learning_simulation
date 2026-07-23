import torch.nn as nn
import torchvision
from loguru import logger
from torchvision.models.resnet import ResNet

from afl_sim.config import ModelConfig
from afl_sim.enums import DatasetType, ModelType
from afl_sim.types import SimulationModel

from .logistic_regression import LogisticRegression
from .simple_cnn import SimpleSequentialCNN


def get_model(dataset: DatasetType, model_config: ModelConfig) -> SimulationModel:
    """
    Initializes and returns a PyTorch model based on the provided configuration.

    Depending on the configuration, this factory creates a Logistic Regression,
    Simple CNN, or a ResNet18 model. For ResNet models, it automatically adapts
    the input stem if the dataset image size is 64x64 or smaller, and replaces
    all BatchNorm2d layers with GroupNorm.

    Args:
        dataset (DatasetType): An enumeration or data object containing dataset
            metadata such as `num_channels`, `num_classes`, and `image_size`.
        model_config (ModelConfig): Configuration object specifying model
            parameters, including the `model_name`.

    Returns:
        SimulationModel: The instantiated PyTorch model adapted to the dataset.
    """
    model_type = model_config.model_name

    match model_type:
        case ModelType.LOG_REG:
            return LogisticRegression(
                in_channels=dataset.num_channels,
                num_classes=dataset.num_classes,
                image_size=dataset.image_size,
            )

        case ModelType.CNN:
            return SimpleSequentialCNN(
                in_channels=dataset.num_channels,
                num_classes=dataset.num_classes,
                image_size=dataset.image_size,
            )

        case ModelType.RESNET18:  # pragma: no branch
            return _resnet_adapted_to_dataset(dataset)


def _resnet_adapted_to_dataset(dataset: DatasetType) -> ResNet:
    """
    Initializes and adapts a ResNet18 model for a specific dataset.

    This function performs three main adaptations on a standard, uninitialized
    ResNet18 model:
    1. Replaces the final fully connected layer to match the dataset's number
       of classes.
    2. If the dataset's image size is 64x64 or smaller, replaces the initial
       7x7 convolution with a 3x3 convolution and replaces the max pooling
       layer with an identity map to prevent excessive downsampling of the input.
    3. Recursively replaces all BatchNorm2d layers with GroupNorm layers.

    Args:
        dataset (DatasetType): An enumeration or data object containing dataset
            metadata such as `num_channels`, `num_classes`, and `image_size`.

    Returns:
        ResNet: The adapted ResNet18 model.
    """
    logger.info(f"Initializing ResNet18 for dataset '{dataset}'...")
    model = torchvision.models.resnet18(weights=None)

    # Adapt final classification layer
    model.fc = nn.Linear(
        model.fc.in_features, out_features=dataset.num_classes, bias=True
    )

    # Adapt input for small images
    if dataset.image_size <= 64:
        logger.info(
            "Adapting ResNet18 stem for input size "
            f"({dataset.image_size}x{dataset.image_size})"
        )
        model.conv1 = nn.Conv2d(
            model.conv1.in_channels,
            model.conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    # Replace BatchNorm with GroupNorm to accommodate federated setting
    logger.info("Replacing BatchNorm with GroupNorm in ResNet18...")
    _resnet_bn_to_gn(model)

    return model


def _resnet_bn_to_gn(module: nn.Module) -> None:
    """
    Recursively replaces all BatchNorm2d layers in a module with GroupNorm.

    The number of groups for the GroupNorm layer is dynamically calculated as
    max(1, num_channels // 16) to ensure compatible and stable grouped normalization.
    The modification is performed in place.

    Args:
        module (nn.Module): The PyTorch module (or model) to modify.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(1, num_channels // 16)

            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            _resnet_bn_to_gn(child)
