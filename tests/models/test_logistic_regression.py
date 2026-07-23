import pytest
import torch
import torch.nn as nn

from afl_sim.models.logistic_regression import LogisticRegression


@pytest.mark.parametrize(
    ("in_channels", "num_classes", "image_size", "batch_size"),
    [
        (1, 10, 28, 1),
        (1, 10, 14, 128),
        (3, 100, 32, 64),
    ],
)
def test_logistic_regression_forward_shape(
    in_channels, num_classes, image_size, batch_size
):
    model = LogisticRegression(in_channels, num_classes, image_size)
    test_tensor = torch.rand(size=(batch_size, in_channels, image_size, image_size))

    output_tensor = model(test_tensor)
    assert output_tensor.shape == (batch_size, num_classes)


def test_logistic_regression_input_shape_mismatch():
    image_size = 28
    model = LogisticRegression(in_channels=1, num_classes=10, image_size=image_size)

    mismatched_tensor = torch.rand(size=(4, 1, 32, 32))

    with pytest.raises(RuntimeError):
        model(mismatched_tensor)


def test_logistic_regression_backward_pass():
    model = LogisticRegression(in_channels=3, num_classes=10, image_size=28)
    test_tensor = torch.rand(size=(4, 3, 28, 28))
    target_labels = torch.randint(0, 10, (4,))

    criterion = nn.CrossEntropyLoss()

    output = model(test_tensor)
    loss = criterion(output, target_labels)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.sum(torch.abs(param.grad)) > 0, (
                f"Gradient is entirely zero for {name}"
            )
