import pytest
import torch

from auto_infer.pd.connector import copy_blocks, transfer_hccl


def test_copy_blocks_copies_only_selected_dense_cache_blocks():
    source = torch.arange(24).reshape(2, 3, 4)
    target = torch.zeros_like(source)

    copy_blocks([source], [target], [1])

    assert torch.equal(target[:, 1], source[:, 1])
    assert not target[:, 0].any()
    assert not target[:, 2].any()


def test_copy_blocks_rejects_layer_count_mismatch():
    with pytest.raises(ValueError, match="layer count"):
        copy_blocks([torch.zeros(2, 1, 1)], [], [0])


def test_transfer_hccl_rejects_unknown_role_before_distributed_io():
    with pytest.raises(ValueError, match="role"):
        transfer_hccl([], "relay", 1)
