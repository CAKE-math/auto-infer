"""Experimental low-level Prefill-to-Decode KV transfer operations.

Transfers the prefill-computed paged KV blocks to the decode instance so decode
continues without recompute. Intra-node transfer copies selected physical blocks;
cross-node transfer sends matching cache tensors over HCCL. These primitives are
an intentionally unwired contract: serving orchestration does not invoke them.
"""


def _cache_tensors(layer):
    return layer if isinstance(layer, tuple) else (layer,)


def transfer_hccl(caches, role, peer, group=None) -> None:
    """Send or receive whole cache tensors over HCCL."""
    if role not in {"producer", "consumer"}:
        raise ValueError("role must be 'producer' or 'consumer'")
    import torch
    import torch.distributed as dist
    for layer in caches:
        for tensor in _cache_tensors(layer):
            if role == "producer":
                dist.send(tensor.contiguous(), dst=peer, group=group)
            else:
                received = torch.empty_like(tensor)
                dist.recv(received, src=peer, group=group)
                tensor.copy_(received)


def copy_blocks(source_caches, target_caches, block_ids) -> None:
    """Copy selected physical blocks between caches with identical layouts.

    A tensor cache uses dense ``(2, blocks, ...)`` layout. A tuple contains
    separate key/value tensors whose leading dimension is the block dimension.
    """
    if len(source_caches) != len(target_caches):
        raise ValueError("source and target cache layer counts must match")
    for source_layer, target_layer in zip(source_caches, target_caches):
        source_tensors = _cache_tensors(source_layer)
        target_tensors = _cache_tensors(target_layer)
        if len(source_tensors) != len(target_tensors):
            raise ValueError("source and target cache layouts must match")
        separate_kv = isinstance(source_layer, tuple)
        for source, target in zip(source_tensors, target_tensors):
            if source.shape != target.shape:
                raise ValueError("source and target cache shapes must match")
            block_axis = 0 if separate_kv else 1
            indices = [slice(None)] * source.ndim
            for block_id in block_ids:
                indices[block_axis] = block_id
                target[tuple(indices)].copy_(source[tuple(indices)])
