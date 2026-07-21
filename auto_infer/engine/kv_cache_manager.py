from collections import OrderedDict


class KVCacheManager:
    """Paged KV block allocator with prefix caching + an evictable LRU pool.

    A block is in exactly one state: free (unused), active (refcount >= 1), or
    cached (refcount 0 but still hash-registered, revivable by match_prefix, and
    evicted LRU-first only when free blocks run out)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._cached: "OrderedDict[int, int]" = OrderedDict()  # block_id -> hash, oldest first
        self._ref: dict[int, int] = {}
        self._hash_to_block: dict[int, int] = {}
        self._block_hash: dict[int, int] = {}
        self._block_tokens: dict[int, tuple] = {}   # blk -> its token chunk (collision guard)
        # prefix-cache hit-rate counters (in blocks), recorded once per admitted
        # request by the scheduler; read by StatLogger. Approximate but stable.
        self.prefix_queried_blocks = 0
        self.prefix_hit_blocks = 0

    def record_prefix_stats(self, queried_blocks: int, hit_blocks: int) -> None:
        self.prefix_queried_blocks += queried_blocks
        self.prefix_hit_blocks += hit_blocks

    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._cached)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks()

    def _evict_one(self) -> int:
        blk, hsh = self._cached.popitem(last=False)     # LRU: oldest freed
        if self._hash_to_block.get(hsh) == blk:
            del self._hash_to_block[hsh]
        self._block_hash.pop(blk, None)
        self._block_tokens.pop(blk, None)
        return blk

    def _alloc_one(self) -> int:
        blk = self._free.pop() if self._free else self._evict_one()
        self._ref[blk] = 1
        return blk

    def allocate(self, num_tokens: int) -> list[int]:
        n = self.blocks_needed(num_tokens)
        if n > self.num_free_blocks():
            raise MemoryError(f"need {n} blocks, {self.num_free_blocks()} free")
        return [self._alloc_one() for _ in range(n)]

    def append_slots(self, block_ids: list[int], cur_num_tokens: int,
                     num_new_tokens: int) -> list[int]:
        total_needed = self.blocks_needed(cur_num_tokens + num_new_tokens)
        extra = total_needed - len(block_ids)
        if extra <= 0:
            return []
        if extra > self.num_free_blocks():
            raise MemoryError(f"need {extra} more blocks, {self.num_free_blocks()} free")
        new = [self._alloc_one() for _ in range(extra)]
        block_ids.extend(new)
        return new

    def free(self, block_ids: list[int]) -> None:
        for b in block_ids:
            if b not in self._ref:
                continue
            self._ref[b] -= 1
            if self._ref[b] <= 0:
                del self._ref[b]
                if b in self._block_hash:            # registered full block -> keep cached
                    self._cached[b] = self._block_hash[b]
                else:
                    self._free.append(b)

    # ---------------- prefix caching (spec sec 4) ----------------
    @staticmethod
    def _block_content_hash(prev_hash: int, chunk: tuple[int, ...]) -> int:
        return hash((prev_hash, chunk))

    def match_prefix(self, token_ids: list[int]) -> list[int]:
        """Return cached physical blocks covering the longest full-block prefix
        of token_ids, incrementing their refcount (shared) or reviving them from
        the evictable pool. COW happens on the first write to a shared block
        (caller copies before mutating)."""
        bs = self.block_size
        matched: list[int] = []
        prev = 0
        for start in range(0, (len(token_ids) // bs) * bs, bs):
            chunk = tuple(token_ids[start:start + bs])
            h = self._block_content_hash(prev, chunk)
            blk = self._hash_to_block.get(h)
            if blk is None:
                break
            if self._block_tokens.get(blk) != chunk:  # hash collision -> tokens differ: MISS
                break                                  # (avoid returning a wrong physical block)
            if blk in self._cached:                  # revive freed-but-cached block
                del self._cached[blk]
                self._ref[blk] = 1
            elif blk in self._ref:                   # shared with a live request
                self._ref[blk] += 1
            else:
                break                                # stale mapping; stop matching
            matched.append(blk)
            prev = h
        return matched

    def register_prefix(self, token_ids: list[int], block_ids: list[int]) -> None:
        """Register completed full blocks for reuse by future requests."""
        bs = self.block_size
        prev = 0
        for idx in range(len(token_ids) // bs):
            chunk = tuple(token_ids[idx * bs:(idx + 1) * bs])
            h = self._block_content_hash(prev, chunk)
            blk = block_ids[idx]
            self._hash_to_block.setdefault(h, blk)
            self._block_hash.setdefault(blk, h)
            self._block_tokens[blk] = chunk          # token-id verify on future match_prefix
            prev = h
