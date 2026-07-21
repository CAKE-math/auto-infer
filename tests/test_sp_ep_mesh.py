"""SP×EP device-mesh planner (spec §6 "attention-DP + MoE-EP 混合切") — host test.
Verifies the two mesh axes are ORTHOGONAL: every (sp_coord, ep_coord) pair maps to
exactly one rank, EP groups share tokens (same sp coord) while splitting experts,
SP groups share experts (same ep coord) while splitting tokens. This orthogonality
is exactly what lets SP token-sharding and EP expert-sharding compose without
colliding (the reason single-axis EP-via-TP could not run SP+EP together)."""
from auto_infer.distributed.mesh import ParallelMesh


def test_2x2_mesh_orthogonal():
    ws = 4
    mesh = ParallelMesh(sp=2, ep=2)
    ep_groups = mesh.groups("ep")
    sp_groups = mesh.groups("sp")
    assert sorted(map(sorted, ep_groups)) == [[0, 1], [2, 3]]   # same tokens, split experts
    assert sorted(map(sorted, sp_groups)) == [[0, 2], [1, 3]]   # same experts, split tokens
    # orthogonality: each rank in exactly one EP group and one SP group; the pair
    # (sp_group_id, ep_group_id) uniquely identifies the rank
    seen = {}
    for r in range(ws):
        epg = next(i for i, g in enumerate(ep_groups) if r in g)
        spg = next(i for i, g in enumerate(sp_groups) if r in g)
        seen[(spg, epg)] = r
    assert len(seen) == ws                       # bijection: 4 distinct (sp,ep) coords


def test_2x4_cover_and_disjoint():
    ws = 8
    mesh = ParallelMesh(sp=2, ep=4)
    ep_groups = mesh.groups("ep")
    sp_groups = mesh.groups("sp")
    assert len(ep_groups) == 2 and all(len(g) == 4 for g in ep_groups)   # 4 experts/group
    assert len(sp_groups) == 4 and all(len(g) == 2 for g in sp_groups)   # 2 token-shards/group
    for groups in (ep_groups, sp_groups):
        flat = sorted(r for g in groups for r in g)
        assert flat == list(range(ws))           # complete + disjoint cover


if __name__ == "__main__":
    test_2x2_mesh_orthogonal()
    test_2x4_cover_and_disjoint()
    print("ALL PASS: SP×EP mesh axes orthogonal (compose), complete + disjoint")
