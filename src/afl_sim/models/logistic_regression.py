from typing import cast

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
    A simple logistic regression model implemented as a single linear layer.

    Flattens the input tensor and passes it through a fully connected layer
    to produce unnormalized class logits.
    """

    def __init__(self, in_channels: int, num_classes: int, image_size: int):
        """
        Initializes the logistic regression model.

        Args:
            in_channels (int): The number of color channels in the input images.
            num_classes (int): The total number of target classes for classification.
            image_size (int): The spatial dimension (height and width) of the square input images.
        """
        super().__init__()
        self.linear = nn.Linear(image_size * image_size * in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): A batch of input image tensors with shape
                (batch_size, in_channels, image_size, image_size).

        Returns:
            torch.Tensor: The output logits with shape (batch_size, num_classes).
        """
        x = x.flatten(1)
        return cast("torch.Tensor", self.linear(x))
