import pytest
from auto_infer.engine.kv_cache_manager import KVCacheManager


def test_blocks_needed():
    m = KVCacheManager(num_blocks=10, block_size=4)
    assert m.blocks_needed(0) == 0
    assert m.blocks_needed(1) == 1
    assert m.blocks_needed(4) == 1
    assert m.blocks_needed(5) == 2


def test_allocate_and_free():
    m = KVCacheManager(num_blocks=10, block_size=4)
    blocks = m.allocate(6)            # needs 2 blocks
    assert len(blocks) == 2
    assert m.num_free_blocks() == 8
    m.free(blocks)
    assert m.num_free_blocks() == 10


def test_append_slots_only_when_last_block_full():
    m = KVCacheManager(num_blocks=10, block_size=4)
    blocks = m.allocate(4)            # exactly 1 full block
    assert m.num_free_blocks() == 9
    # currently 4 tokens, add 1 -> needs a new block
    new = m.append_slots(blocks, cur_num_tokens=4, num_new_tokens=1)
    assert len(new) == 1
    assert m.num_free_blocks() == 8
    # currently 5 tokens (room for 3 more in 2nd block), add 2 -> no new block
    new2 = m.append_slots(blocks, cur_num_tokens=5, num_new_tokens=2)
    assert new2 == []


def test_allocate_oom():
    m = KVCacheManager(num_blocks=1, block_size=4)
    with pytest.raises(MemoryError):
        m.allocate(100)


def test_can_allocate():
    m = KVCacheManager(num_blocks=2, block_size=4)
    assert m.can_allocate(8) is True
    assert m.can_allocate(9) is False


def test_prefix_caching_share_and_release():
    m = KVCacheManager(num_blocks=10, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8]          # 2 full blocks
    blk = m.allocate(8)
    m.register_prefix(toks, blk)
    free_after_first = m.num_free_blocks()    # 8 free
    # second request with same prefix reuses both blocks (refcount++)
    shared = m.match_prefix(toks)
    assert shared == blk
    assert m.num_free_blocks() == free_after_first   # no new physical blocks
    # releasing the shared refs does not return blocks while first still holds
    m.free(shared)
    assert m.num_free_blocks() == free_after_first
    # releasing the original returns them and clears the prefix cache
    m.free(blk)
    assert m.num_free_blocks() == 10
    assert m.match_prefix(toks) == blk         # freed blocks stay cached & revive


def test_prefix_partial_match():
    m = KVCacheManager(num_blocks=10, block_size=4)
    blk = m.allocate(8)
    m.register_prefix([1, 2, 3, 4, 5, 6, 7, 8], blk)
    # different second block -> only first block matches
    got = m.match_prefix([1, 2, 3, 4, 9, 9, 9, 9])
    assert got == [blk[0]]


def test_free_retains_registered_block_for_reuse():
    m = KVCacheManager(num_blocks=10, block_size=4)
    toks = [1, 2, 3, 4, 5, 6, 7, 8]           # 2 full blocks
    blk = m.allocate(8)
    m.register_prefix(toks, blk)
    m.free(blk)                                # refcount 0 -> stays in evictable cache
    assert m.num_free_blocks() == 10           # counted as allocatable
    revived = m.match_prefix(toks)             # cache HIT after free (new behavior)
    assert revived == blk
    assert m.num_free_blocks() == 8            # revived blocks now held (ref=1)


def test_lru_eviction_unregisters_oldest():
    m = KVCacheManager(num_blocks=2, block_size=4)
    a = m.allocate(4)                          # block for prefix A
    m.register_prefix([1, 2, 3, 4], a)
    m.free(a)                                  # A cached (1 free real + 1 cached)
    # allocate 2 blocks: consumes the 1 real free + evicts cached A
    two = m.allocate(8)
    assert len(two) == 2
    assert m.num_free_blocks() == 0
    # A's hash was unregistered on eviction -> no longer matchable
    assert m.match_prefix([1, 2, 3, 4]) == []


def test_lru_eviction_order_two_blocks():
    """Discriminating test: with two distinct cached blocks, the OLDER one
    (A) must be evicted before the NEWER one (B). A test with only a single
    cached block cannot tell LRU-first eviction apart from any other order."""
    m = KVCacheManager(num_blocks=3, block_size=4)
    a = m.allocate(4)                          # block for prefix A
    m.register_prefix([1, 2, 3, 4], a)
    m.free(a)                                  # A cached first -> oldest (free=2, cached=1)
    b = m.allocate(4)                          # consumes the 1 remaining real free block
    m.register_prefix([5, 6, 7, 8], b)
    m.free(b)                                  # B cached second -> newest (free=1, cached=2)
    assert m.num_free_blocks() == 3
    # allocate 2 blocks: consumes the 1 real free block, then forces exactly
    # one eviction from the cache (A and B are the only candidates)
    two = m.allocate(8)
    assert len(two) == 2
    # A (oldest cached) must be the one evicted -> no longer matchable
    assert m.match_prefix([1, 2, 3, 4]) == []
    # B (newest cached) must still be revivable -> proves eviction was LRU-first
    assert m.match_prefix([5, 6, 7, 8]) == b
