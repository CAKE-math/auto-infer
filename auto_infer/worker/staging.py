"""Shared pinned-host allocation and dirty-row utilities for graph staging."""
import numpy as np
import torch


class HostStaging:
    def __init__(self, pin_memory: bool):
        self.non_blocking = pin_memory

    def allocate(self, shape, dtype):
        tensor = torch.empty(shape, dtype=dtype)
        if self.non_blocking:
            try:
                tensor = tensor.pin_memory()
            except (RuntimeError, NotImplementedError):
                self.non_blocking = False
        return tensor, tensor.numpy()


def dirty_spans(dirty):
    start = None
    for index, changed in enumerate(dirty.tolist() + [False]):
        if changed and start is None:
            start = index
        elif not changed and start is not None:
            yield start, index
            start = None


def upload_dirty_block_table(
    device,
    host,
    current,
    shadow,
    non_blocking,
    *,
    active_rows=None,
):
    """Upload changed block-table rows and return row/element copy counts."""
    dirty = np.any(current != shadow, axis=1)
    if active_rows is not None:
        dirty[active_rows:] = False
    copied_rows = 0
    for start, end in dirty_spans(dirty):
        device[start:end].copy_(host[start:end], non_blocking=non_blocking)
        shadow[start:end] = current[start:end]
        copied_rows += end - start
    return copied_rows, copied_rows * current.shape[1]


def splice_device_tokens(target, target_rows, request_order, refs):
    """Splice retained samples in owner-sized vector operations, never clones."""
    groups = {}
    for target_row, request_id in zip(target_rows, request_order):
        ref = refs.get(request_id)
        if ref is None:
            continue
        key = id(ref.owner)
        if key not in groups:
            groups[key] = (ref.owner, [], [])
        groups[key][1].append(target_row)
        groups[key][2].append(ref.row)

    for owner, destinations, sources in groups.values():
        source_rows = torch.as_tensor(
            sources, dtype=torch.long, device=owner.tokens.device)
        target_indices = torch.as_tensor(
            destinations, dtype=torch.long, device=target.device)
        values = owner.tokens.index_select(0, source_rows)
        target.index_copy_(0, target_indices, values)
