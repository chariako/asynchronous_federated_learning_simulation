import filecmp
import os

import torch


def valid_tensor_dict() -> dict[str, torch.Tensor]:
    return {"weights": torch.rand(size=(2,))}


def assert_tensor_dicts_equal(
    dict1: dict[str, torch.Tensor], dict2: dict[str, torch.Tensor]
):
    assert dict1.keys() == dict2.keys(), "Dictionary keys do not match."
    for key in dict1:
        assert torch.equal(dict1[key], dict2[key]), (
            f"Tensors for key '{key}' do not match."
        )


def assert_copied_files(src_file, dst_file):
    assert filecmp.cmp(src_file, dst_file, shallow=False)

    stat_src = os.stat(src_file)
    stat_dst = os.stat(dst_file)

    assert stat_src.st_mtime == stat_dst.st_mtime

    assert stat_src.st_mode == stat_dst.st_mode
