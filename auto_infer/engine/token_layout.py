def slot_mapping(block_table, positions, block_size):
    """Map logical positions to paged-cache slots (scalar or NumPy array)."""
    return (block_table[positions // block_size] * block_size
            + positions % block_size)
