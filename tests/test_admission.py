import pytest

from auto_infer.serving.admission import (AdmissionController, Overloaded,
                                          Unavailable)


def test_admission_has_hard_request_and_token_limits():
    gate = AdmissionController(
        max_http=2, max_engine_requests=1, max_engine_tokens=10
    )
    http = gate.acquire_http()
    engine = gate.acquire_engine(prompt_tokens=8)

    with pytest.raises(Overloaded, match="request capacity"):
        gate.acquire_engine(prompt_tokens=1)

    engine.release()
    replacement = gate.acquire_engine(prompt_tokens=10)
    replacement.release()
    http.release()

    snapshot = gate.snapshot()
    assert snapshot.http_inflight == 0
    assert snapshot.engine_requests == 0
    assert snapshot.engine_tokens == 0
    assert snapshot.permits_in_use == 0


def test_engine_token_limit_rejects_without_consuming_request_slot():
    gate = AdmissionController(
        max_http=1, max_engine_requests=2, max_engine_tokens=10
    )

    with pytest.raises(Overloaded, match="token capacity"):
        gate.acquire_engine(prompt_tokens=11)

    assert gate.snapshot().engine_requests == 0


def test_admission_lease_release_is_idempotent():
    gate = AdmissionController(
        max_http=1, max_engine_requests=1, max_engine_tokens=10
    )
    lease = gate.acquire_http()

    lease.release()
    lease.release()

    assert gate.snapshot().permits_in_use == 0


def test_closed_admission_rejects_and_can_reopen_after_recovery():
    gate = AdmissionController(
        max_http=1, max_engine_requests=1, max_engine_tokens=10
    )
    gate.close()

    with pytest.raises(Unavailable, match="closed"):
        gate.acquire_http()

    gate.open()
    lease = gate.acquire_http()
    lease.release()
