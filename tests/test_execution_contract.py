from dataclasses import FrozenInstanceError

import pytest

from auto_infer.config import SchedulerConfig
from auto_infer.engine.execution import BatchPlan, ExecutionResult
from auto_infer.engine.executor import MockExecutor
from auto_infer.engine.kv_cache_manager import KVCacheManager
from auto_infer.engine.request import Request, SamplingParams
from auto_infer.engine.scheduler import Scheduler


def _plan():
    scheduler = Scheduler(
        SchedulerConfig(max_num_batched_tokens=16),
        KVCacheManager(num_blocks=16, block_size=4),
    )
    request = Request("r", [1, 2, 3], SamplingParams(max_tokens=2))
    scheduler.add_request(request)
    output = scheduler.schedule()
    return scheduler, request, BatchPlan.from_scheduler(output, scheduler)


def test_batch_plan_metadata_is_immutable_and_token_lengths_are_snapshotted():
    _, request, plan = _plan()
    view = plan.get_request("r")
    with pytest.raises(FrozenInstanceError):
        view.num_computed_tokens = 9
    request.output_token_ids.append(99)
    assert view.output_token_ids == ()
    assert view.prompt_token_ids == (1, 2, 3)


def test_batch_plan_token_view_references_request_storage_without_growth_leak():
    _, request, _ = _plan()
    request.output_token_ids.append(4)
    output = request.output_token_ids
    # Token views snapshot lengths while retaining the underlying storage.
    from auto_infer.engine.execution import TokenView
    view = TokenView(request.prompt_token_ids, output)
    request.output_token_ids.append(5)

    assert view[:] == [1, 2, 3, 4]
    assert view.output_source is output


def test_mock_executor_consumes_plan_and_returns_execution_result():
    _, _, plan = _plan()
    result = MockExecutor(vocab_size=100).execute(plan)
    assert isinstance(result, ExecutionResult)
    assert result.tokens == {"r": (4,)}
