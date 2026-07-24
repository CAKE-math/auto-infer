from auto_infer.distributed import parallel_state


def test_tp_barrier_is_noop_for_single_rank(monkeypatch):
    called = []
    monkeypatch.setattr(parallel_state, "_TP_SIZE", 1)
    monkeypatch.setattr(
        "torch.distributed.barrier", lambda **kwargs: called.append(kwargs)
    )

    parallel_state.tp_barrier()

    assert called == []


def test_tp_barrier_uses_tensor_parallel_group(monkeypatch):
    group = object()
    called = []
    monkeypatch.setattr(parallel_state, "_TP_SIZE", 2)
    monkeypatch.setattr(parallel_state, "_TP_GROUP", group)
    monkeypatch.setattr(
        "torch.distributed.barrier", lambda **kwargs: called.append(kwargs)
    )

    parallel_state.tp_barrier()

    assert called == [{"group": group}]
