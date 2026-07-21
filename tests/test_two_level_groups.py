"""§6 two-level topology comm-group planning — host-only unit test (no NPU/NICs).
Verifies intra-node (HCCS) and inter-node (RDMA) group partitions for a simulated
multi-node layout: complete cover, correct membership, expected DeepSeek-V3 mapping
(TP intra-node, EP inter-node)."""
from auto_infer.distributed.parallel_state import plan_two_level_groups


def test_2node_4proc():
    plan = plan_two_level_groups(nnodes=2, nproc_per_node=4)
    assert plan["world_size"] == 8
    assert plan["intra"] == [[0, 1, 2, 3], [4, 5, 6, 7]]        # TP/dense over HCCS
    assert plan["inter"] == [[0, 4], [1, 5], [2, 6], [3, 7]]    # EP all-to-all over RDMA


def test_cover_and_disjoint():
    plan = plan_two_level_groups(nnodes=4, nproc_per_node=8)
    ws = plan["world_size"]
    assert ws == 32
    # every rank in exactly one intra group and exactly one inter group
    for ranks in (plan["intra"], plan["inter"]):
        seen = [r for g in ranks for r in g]
        assert sorted(seen) == list(range(ws))
        assert len(seen) == len(set(seen))
    # each intra group is one node (contiguous block of nproc_per_node)
    assert all(len(g) == 8 for g in plan["intra"]) and len(plan["intra"]) == 4
    # each inter group has one rank per node
    assert all(len(g) == 4 for g in plan["inter"]) and len(plan["inter"]) == 8


def test_single_node_degenerate():
    plan = plan_two_level_groups(nnodes=1, nproc_per_node=8)
    assert plan["intra"] == [list(range(8))]
    assert plan["inter"] == [[i] for i in range(8)]             # no cross-node pairing


if __name__ == "__main__":
    test_2node_4proc()
    test_cover_and_disjoint()
    test_single_node_degenerate()
    print("ALL PASS: two-level topology group planning (intra HCCS / inter RDMA)")
