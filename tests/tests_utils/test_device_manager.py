import pytest

from afl_sim.enums import DeviceType
from afl_sim.utils import get_device


@pytest.mark.parametrize(
    "device_type, check_func_path, expected_device",
    [
        (DeviceType.CUDA, "torch.cuda.is_available", "cuda"),
        (DeviceType.MPS, "torch.backends.mps.is_available", "mps"),
    ],
)
def test_explicit_device_success(
    monkeypatch, device_type, check_func_path, expected_device
):
    monkeypatch.setattr(check_func_path, lambda: True)
    device = get_device(device_type)
    assert device.type == expected_device


@pytest.mark.parametrize(
    "device_type, check_func_path",
    [
        (DeviceType.CUDA, "torch.cuda.is_available"),
        (DeviceType.MPS, "torch.backends.mps.is_available"),
    ],
)
def test_explicit_device_failure(monkeypatch, device_type, check_func_path):
    monkeypatch.setattr(check_func_path, lambda: False)
    with pytest.raises(ValueError, match="requested but not available"):
        get_device(device_type)


def test_cpu_device_handling():
    device = get_device(DeviceType.CPU)
    assert device.type == "cpu"


@pytest.mark.parametrize(
    "cuda_exists, mps_exists, returned_device",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_handling(monkeypatch, cuda_exists, mps_exists, returned_device):
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda_exists)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps_exists)
    device = get_device(DeviceType.AUTO)
    assert device.type == returned_device
