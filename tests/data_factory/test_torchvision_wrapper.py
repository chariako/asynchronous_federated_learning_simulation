import numpy as np
import pytest
from torchvision.transforms import v2

from afl_sim.data_factory.torchvision_wrapper import TorchvisionDatasetWrapper
from afl_sim.enums import DatasetType

MODULEPATH = "afl_sim.data_factory.torchvision_wrapper.TorchvisionDatasetWrapper"


def _get_data_class_test_cases() -> list[tuple[DatasetType, str]]:
    cases = []
    for item in list(DatasetType):
        if item.source == "torchvision":
            cases.append((item, item.source_name))
    return cases


@pytest.mark.parametrize(
    ("dataset_type", "dataset_class"),
    _get_data_class_test_cases(),
)
def test_dataset_assignment(mocker, tmp_path, dataset_type, dataset_class):
    mocker.patch(f"{MODULEPATH}._build_train_transform_list", return_value=None)
    mocker.patch(f"{MODULEPATH}._build_eval_transform_list", return_value=None)
    mock_data_load = mocker.patch(
        f"afl_sim.data_factory.torchvision_wrapper.{dataset_class}"
    )
    TorchvisionDatasetWrapper(dataset_type=dataset_type, data_root=tmp_path)

    assert mock_data_load.call_count == 2

    train_kwargs = mock_data_load.call_args_list[0].kwargs
    assert train_kwargs["train"] is True
    assert train_kwargs["download"] is True
    assert isinstance(train_kwargs["transform"], v2.ToImage)

    eval_kwargs = mock_data_load.call_args_list[1].kwargs
    assert eval_kwargs["train"] is False
    assert isinstance(eval_kwargs["transform"], v2.ToImage)


def _get_train_transform_test_cases() -> list[tuple[DatasetType, v2.Transform]]:
    cases = []
    for item in list(DatasetType):
        if item.source == "torchvision":
            transform_list = []
            if item.apply_horizontal_flip_transform:
                transform_list.append(v2.RandomHorizontalFlip)
            if item.apply_crop_transform:
                transform_list.append(v2.RandomCrop)
            transform_list.append(v2.ToDtype)
            transform_list.append(v2.Normalize)

            cases.append((item, transform_list))
    return cases


@pytest.mark.parametrize(
    ("dataset_type", "expected_transforms"),
    _get_train_transform_test_cases(),
)
def test_transform_builders(mocker, tmp_path, dataset_type, expected_transforms):
    mocker.patch(f"{MODULEPATH}._load_data", return_value=None)
    mocker.patch(f"{MODULEPATH}._build_eval_transform_list", return_value=None)

    wrapper = TorchvisionDatasetWrapper(dataset_type=dataset_type, data_root=tmp_path)

    tf_list = wrapper._train_transform_list

    for i, tf in enumerate(tf_list):
        assert isinstance(tf, expected_transforms[i])
        if isinstance(tf, v2.Normalize):
            assert list(tf.mean) == list(dataset_type.mean)
            assert list(tf.std) == list(dataset_type.std)
        if isinstance(tf, v2.RandomCrop):
            assert tf.size == (dataset_type.image_size, dataset_type.image_size)
            assert tf.padding == [int(dataset_type.image_size / 8)] * 4


@pytest.mark.parametrize(
    ("dataset_type", "expected_transforms"),
    [
        (item, [v2.ToDtype, v2.Normalize])
        for item in list(DatasetType)
        if item.source == "torchvision"
    ],
)
def test_build_eval_transform(mocker, tmp_path, dataset_type, expected_transforms):
    mocker.patch(f"{MODULEPATH}._load_data", return_value=None)
    mocker.patch(f"{MODULEPATH}._build_train_transform_list", return_value=None)

    wrapper = TorchvisionDatasetWrapper(dataset_type=dataset_type, data_root=tmp_path)

    tf_list = wrapper._eval_transform_list

    for i, tf in enumerate(tf_list):
        assert isinstance(tf, expected_transforms[i])
        if isinstance(tf, v2.Normalize):
            assert list(tf.mean) == list(dataset_type.mean)
            assert list(tf.std) == list(dataset_type.std)


def test_train_subset(mocker, tmp_path):
    mocker.patch(f"{MODULEPATH}._load_data", return_value=None)
    mocker.patch(f"{MODULEPATH}._build_train_transform_list", return_value=None)
    mocker.patch(f"{MODULEPATH}._build_eval_transform_list", return_value=None)
    mock_subset_call = mocker.patch("afl_sim.data_factory.torchvision_wrapper.Subset")

    wrapper = TorchvisionDatasetWrapper(
        dataset_type=DatasetType.MNIST, data_root=tmp_path
    )
    wrapper._train_data = None
    indices = np.array([1, 2, 3], dtype=np.int64)
    wrapper.get_subset(indices)

    mock_subset_call.assert_called_once_with(wrapper._train_data, indices)
