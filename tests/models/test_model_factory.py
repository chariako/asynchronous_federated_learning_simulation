import pytest
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet

from afl_sim.config import ModelConfig
from afl_sim.enums import DatasetType, ModelType
from afl_sim.models.logistic_regression import LogisticRegression
from afl_sim.models.model_factory import (
    _resnet_adapted_to_dataset,
    _resnet_bn_to_gn,
    get_model,
)
from afl_sim.models.simple_cnn import SimpleSequentialCNN

MODEL_TYPE_MAPPING = {
    ModelType.LOG_REG: LogisticRegression,
    ModelType.CNN: SimpleSequentialCNN,
    ModelType.RESNET18: ResNet,
}


def _get_model_type_test_cases() -> list[tuple[ModelConfig, type[nn.Module]]]:
    cases = []
    for model_type in ModelType:
        if model_type not in MODEL_TYPE_MAPPING:
            raise ValueError(
                f"Model Factory Test Error: '{model_type}' missing return type mapping."
            )

        cases.append(
            (ModelConfig(model_name=model_type), MODEL_TYPE_MAPPING[model_type])
        )
    return cases


@pytest.mark.parametrize(
    ("model_config", "expected_output_type"), _get_model_type_test_cases()
)
def test_get_model_returns_correct_type(model_config, expected_output_type, mocker):
    dataset = DatasetType.CIFAR10

    match model_config.model_name:
        case ModelType.CNN | ModelType.LOG_REG:
            spy = mocker.spy(expected_output_type, "__init__")
        case ModelType.RESNET18:
            mock_resnet_load = mocker.patch(
                "afl_sim.models.model_factory._resnet_adapted_to_dataset",
                return_value=torchvision.models.resnet18(weights=None),
            )

    model = get_model(dataset=dataset, model_config=model_config)

    assert isinstance(model, expected_output_type)

    match model_config.model_name:
        case ModelType.CNN | ModelType.LOG_REG:
            spy.assert_called_once_with(
                mocker.ANY,
                in_channels=dataset.num_channels,
                num_classes=dataset.num_classes,
                image_size=dataset.image_size,
            )
        case ModelType.RESNET18:
            mock_resnet_load.assert_called_once_with(dataset)


def _get_batchnorm_list(model) -> list[dict[str, int]]:
    bn_list = [
        {"num_channels": module.num_features}
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    return bn_list


def _get_groupnorm_list(model) -> list[dict[str, int]]:
    gn_list = [
        {"num_channels": module.num_channels, "num_groups": module.num_groups}
        for module in model.modules()
        if isinstance(module, nn.GroupNorm)
    ]
    return gn_list


def test_replace_resnet_batchnorm_with_groupnorm():
    resnet = torchvision.models.resnet18(weights=None)
    bn_list = _get_batchnorm_list(resnet)
    assert bn_list

    _resnet_bn_to_gn(resnet)
    gn_list = _get_groupnorm_list(resnet)

    assert gn_list
    assert _get_batchnorm_list(resnet) == []

    for gn, bn in zip(gn_list, bn_list, strict=True):
        assert gn["num_channels"] == bn["num_channels"]
        assert gn["num_groups"] == max(1, bn["num_channels"] // 16)


@pytest.mark.parametrize("dataset", [DatasetType.CIFAR10, DatasetType.CIFAR100])
def test_resnet_adaptation(dataset):
    adapted_resnet = _resnet_adapted_to_dataset(dataset=dataset)

    assert adapted_resnet.fc.out_features == dataset.num_classes

    if dataset.image_size <= 64:
        assert adapted_resnet.conv1.kernel_size == (3, 3)
        assert adapted_resnet.conv1.stride == (1, 1)
        assert adapted_resnet.conv1.padding == (1, 1)

        assert isinstance(adapted_resnet.maxpool, nn.Identity)


def test_adapted_resnet_supports_forward_pass():
    dataset = DatasetType.CIFAR10
    config = ModelConfig(model_name=ModelType.RESNET18)

    model = get_model(dataset=dataset, model_config=config)

    batch_size = 2
    dummy_input = torch.rand(
        size=(batch_size, dataset.num_channels, dataset.image_size, dataset.image_size)
    )

    output = model(dummy_input)

    assert output.shape == (batch_size, dataset.num_classes)
