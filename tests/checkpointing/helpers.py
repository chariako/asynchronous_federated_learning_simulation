import filecmp
import os

import torch


def valid_tensor_dict() -> dict[str, torch.Tensor]:
    return {"weights": torch.rand(size=(2,))}


def assert_copied_files(src_file, dst_file) -> None:
    assert filecmp.cmp(src_file, dst_file, shallow=False)

    stat_src = os.stat(src_file)
    stat_dst = os.stat(dst_file)

    assert stat_src.st_mtime == stat_dst.st_mtime

    assert stat_src.st_mode == stat_dst.st_mode
