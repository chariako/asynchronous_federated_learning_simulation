from typing import cast

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    A foundational convolutional block for feature extraction.

    Consists of a 2D convolution (without bias), followed by group normalization,
    an in-place ReLU activation, and 2D max pooling to halve spatial dimensions.
    Uses GroupNorm by default to accommodate federated learning.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the convolutional block.

        Args:
            in_channels (int): The number of input channels for the convolutional layer.
            out_channels (int): The number of output feature maps produced by the convolution.
        """
        super().__init__()

        # Using GroupNorm by default
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the convolutional block.

        Args:
            x (torch.Tensor): Input tensor batch with shape
                (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: The processed feature maps with shape
                (batch_size, out_channels, height // 2, width // 2).
        """
        return cast("torch.Tensor", self.block(x))


class SimpleSequentialCNN(nn.Module):
    """
    A straightforward sequential Convolutional Neural Network.

    Extracts spatial features using three sequential `CNNBlock` layers, automatically
    computes the required flattening dimension, and applies a multi-layer perceptron
    with dropout for final classification.
    """

    def __init__(self, in_channels: int, num_classes: int, image_size: int):
        """
        Initializes the sequential CNN model.

        Args:
            in_channels (int): The number of color channels in the input images.
            num_classes (int): The total number of target classes for classification.
            image_size (int): The spatial dimension (height and width) of the square input images.
        """

        super().__init__()

        self.features = nn.Sequential(
            CNNBlock(in_channels, 32),
            CNNBlock(32, 64),
            CNNBlock(64, 64),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, image_size, image_size)
            flat_size = self.features(dummy_input).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the complete network.

        Args:
            x (torch.Tensor): A batch of input image tensors with shape
                (batch_size, in_channels, image_size, image_size).

        Returns:
            torch.Tensor: The output logits with shape (batch_size, num_classes).
        """
        x = self.features(x)
        return cast("torch.Tensor", self.classifier(x))
