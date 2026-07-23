import pytest
import torch
import torch.nn as nn

from afl_sim.models.simple_cnn import CNNBlock, SimpleSequentialCNN


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "batch_size"),
    [
        (1, 32, 1),
        (3, 32, 128),
        (32, 64, 128),
    ],
)
def test_cnn_block(in_channels, out_channels, batch_size):
    cnn_block = CNNBlock(in_channels, out_channels)
    image_size = 28
    test_tensor = torch.rand(size=(batch_size, in_channels, image_size, image_size))

    output_tensor = cnn_block(test_tensor)

    assert output_tensor.shape[0] == batch_size
    assert output_tensor.shape[1] == out_channels
    assert output_tensor.shape[2] == image_size // 2
    assert output_tensor.shape[3] == image_size // 2


@pytest.mark.parametrize(
    ("in_channels", "num_classes", "batch_size"),
    [
        (1, 10, 1),
        (3, 100, 128),
    ],
)
def test_simple_cnn(in_channels, num_classes, batch_size):
    image_size = 28
    model = SimpleSequentialCNN(in_channels, num_classes, image_size)

    test_tensor = torch.rand(size=(batch_size, in_channels, image_size, image_size))
    output_tensor = model(test_tensor)

    assert output_tensor.shape[0] == batch_size
    assert output_tensor.shape[1] == num_classes


def test_cnn_block_invalid_groupnorm_channels():
    in_channels = 3
    invalid_out_channels = 10

    with pytest.raises(ValueError, match="must be divisible by num_groups"):
        CNNBlock(in_channels, invalid_out_channels)


def test_simple_cnn_image_too_small():
    in_channels = 3
    num_classes = 10
    invalid_image_size = 7

    with pytest.raises(RuntimeError):
        SimpleSequentialCNN(in_channels, num_classes, invalid_image_size)


def test_simple_cnn_backward_pass():
    model = SimpleSequentialCNN(in_channels=3, num_classes=10, image_size=28)
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


def test_simple_cnn_train_vs_eval_mode():
    model = SimpleSequentialCNN(in_channels=3, num_classes=10, image_size=28)
    test_tensor = torch.rand(size=(2, 3, 28, 28))

    model.eval()
    with torch.no_grad():
        eval_output_1 = model(test_tensor)
        eval_output_2 = model(test_tensor)

    assert torch.allclose(eval_output_1, eval_output_2), (
        "Eval mode outputs should be deterministic"
    )

    model.train()
    train_output_1 = model(test_tensor)
    train_output_2 = model(test_tensor)

    assert not torch.allclose(train_output_1, train_output_2), (
        "Train mode outputs should differ due to dropout"
    )
