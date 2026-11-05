from __future__ import annotations

import asyncio
import email.message
import http.server
import inspect
import io
import json
import logging
import threading
import typing as t
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

import ragas.llms.bulk as bulk_module
import ragas.llms.inline_batch as inline_batch_module
from ragas.exceptions import RagasOutputParserException
from ragas.llms import (
    BaseRagasBulkLLM,
    BulkCheckpointMismatchError,
    BulkGenerationError,
    BulkLLMLimits,
    BulkTransportError,
    InlineBatchBulkLLM,
)
from ragas.llms.bulk import (
    DEFAULT_CHECKPOINT_PATH,
    BulkBatchResult,
    BulkCheckpoint,
    BulkRequest,
    BulkRequestState,
    BulkUsage,
    pack_requests,
)
from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.transforms.default import default_transforms_for_prechunked
from ragas.testset.transforms.engine import Parallel
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    SummaryExtractor,
    ThemesExtractor,
)


def _request(cid: str, size_bytes: int = 100, tokens: int = 10) -> BulkRequest:
    return BulkRequest(
        custom_id=cid,
        prompt="",
        body={},
        size_bytes=size_bytes,
        tokens=tokens,
    )


class FakeBulkLLM(BaseRagasBulkLLM):
    """In-memory bulk endpoint. ``store`` plays the role of the remote provider."""

    def __init__(
        self,
        limits: BulkLLMLimits,
        store: t.Dict[str, t.Any],
        fail_counts: t.Optional[t.Dict[str, int]] = None,
        fail_submit_on: t.Optional[int] = None,
        in_progress_polls: int = 0,
        respond: t.Optional[t.Callable[[str], str]] = None,
        **kwargs: t.Any,
    ):
        kwargs.setdefault("checkpoint_path", None)
        super().__init__(limits, poll_interval=0, **kwargs)
        self.store = store
        self.fail_counts = fail_counts or {}
        self.fail_submit_on = fail_submit_on
        self.in_progress_polls = in_progress_polls
        self.respond = respond or (lambda prompt: f"answer:{prompt}")

    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        submitted: t.List[t.List[str]] = self.store.setdefault("submitted", [])
        if self.fail_submit_on is not None and len(submitted) == self.fail_submit_on:
            raise ConnectionError("provider unavailable")
        submitted.append([r.custom_id for r in requests])

        responses: t.Dict[str, str] = {}
        errors: t.Dict[str, str] = {}
        for r in requests:
            if self.fail_counts.get(r.custom_id, 0) > 0:
                self.fail_counts[r.custom_id] -= 1
                errors[r.custom_id] = "rate limited"
            else:
                responses[r.custom_id] = self.respond(r.prompt)

        batch_id = f"batch-{len(submitted)}"
        self.store.setdefault("batches", {})[batch_id] = BulkBatchResult(
            status="completed", responses=responses, errors=errors
        )
        return batch_id

    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        if self.in_progress_polls > 0:
            self.in_progress_polls -= 1
            return BulkBatchResult(status="in_progress")
        return self.store["batches"][batch_id]


def test_limits_reject_non_positive_values():
    with pytest.raises(ValueError):
        BulkLLMLimits(max_requests=0)
    with pytest.raises(ValueError):
        BulkLLMLimits(max_request_mb=-1)


def test_pack_respects_request_count():
    requests = [_request(str(i)) for i in range(5)]
    bins = pack_requests(requests, BulkLLMLimits(max_requests=2))
    assert len(bins) == 3
    assert all(len(b) <= 2 for b in bins)
    assert sorted(r.custom_id for b in bins for r in b) == sorted(
        r.custom_id for r in requests
    )


def test_pack_respects_request_size_in_mb():
    requests = [_request(str(i), size_bytes=400_000) for i in range(5)]
    bins = pack_requests(requests, BulkLLMLimits(max_request_mb=1))
    # 1 MB = 1_048_576 bytes, so two 400 KB requests fit and a third does not.
    assert len(bins) == 3
    for b in bins:
        assert sum(r.size_bytes for r in b) <= 1_048_576


def test_pack_respects_context_tokens():
    requests = [
        _request("a", tokens=60),
        _request("b", tokens=50),
        _request("c", tokens=40),
    ]
    bins = pack_requests(requests, BulkLLMLimits(max_context_tokens=100))
    assert len(bins) == 2
    for b in bins:
        assert sum(r.tokens for r in b) <= 100


def test_pack_rejects_request_larger_than_a_batch():
    with pytest.raises(ValueError):
        pack_requests(
            [_request("big", size_bytes=2_000_000)], BulkLLMLimits(max_request_mb=1)
        )


def test_build_body_and_size_are_computed_from_prompt(tmp_path):
    llm = FakeBulkLLM(BulkLLMLimits(), store={})
    state_body = llm.build_body("hello")
    assert state_body == {"messages": [{"role": "user", "content": "hello"}]}
    assert llm.count_tokens("12345678") == 2


@pytest.mark.asyncio
async def test_generate_bulk_packs_and_returns_aligned_responses():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(BulkLLMLimits(max_requests=2), store=store)
    prompts = [f"p{i}" for i in range(5)]

    responses = await llm.generate_bulk(prompts)

    assert responses == [f"answer:p{i}" for i in range(5)]
    assert len(store["submitted"]) == 3


@pytest.mark.asyncio
async def test_generate_bulk_waits_for_in_progress_batches():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(BulkLLMLimits(), store=store, in_progress_polls=2)

    responses = await llm.generate_bulk(["a", "b"])

    assert responses == ["answer:a", "answer:b"]
    assert len(store["submitted"]) == 1


@pytest.mark.asyncio
async def test_failed_request_is_retried_in_a_new_batch():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(
        BulkLLMLimits(max_requests=10),
        store=store,
        fail_counts={"1": 1},
    )

    responses = await llm.generate_bulk(["a", "b", "c"], custom_ids=["0", "1", "2"])

    assert responses == ["answer:a", "answer:b", "answer:c"]
    assert len(store["submitted"]) == 2
    assert store["submitted"][1] == ["1"]


@pytest.mark.asyncio
async def test_permanent_failure_raises_after_max_attempts():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(
        BulkLLMLimits(),
        store=store,
        fail_counts={"1": 100},
        max_attempts=3,
    )

    with pytest.raises(BulkGenerationError) as exc_info:
        await llm.generate_bulk(["a", "b"], custom_ids=["0", "1"])

    assert "1" in exc_info.value.failed
    assert sum("1" in ids for ids in store["submitted"]) == 3


@pytest.mark.asyncio
async def test_permanent_failure_can_return_none_instead_of_raising():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(
        BulkLLMLimits(), store=store, fail_counts={"1": 100}, max_attempts=2
    )

    responses = await llm.generate_bulk(
        ["a", "b"], custom_ids=["0", "1"], raise_exceptions=False
    )

    assert responses == ["answer:a", None]


@pytest.mark.asyncio
async def test_checkpoint_resumes_after_crash_without_resubmitting(tmp_path):
    checkpoint = tmp_path / "bulk.json"
    store: t.Dict[str, t.Any] = {}
    prompts = [f"p{i}" for i in range(5)]
    ids = [str(i) for i in range(5)]

    crashing = FakeBulkLLM(
        BulkLLMLimits(max_requests=2),
        store=store,
        checkpoint_path=checkpoint,
        fail_submit_on=1,
    )
    with pytest.raises(ConnectionError):
        await crashing.generate_bulk(prompts, custom_ids=ids)

    first_run_ids = list(store["submitted"][0])
    saved = BulkCheckpoint.load(checkpoint)
    assert len(saved.batches) == 1

    resumed = FakeBulkLLM(
        BulkLLMLimits(max_requests=2), store=store, checkpoint_path=checkpoint
    )
    responses = await resumed.generate_bulk(prompts, custom_ids=ids)

    assert responses == [f"answer:p{i}" for i in range(5)]
    later_ids = [cid for batch in store["submitted"][1:] for cid in batch]
    assert set(later_ids).isdisjoint(first_run_ids)
    assert sorted(later_ids + first_run_ids) == sorted(ids)


@pytest.mark.asyncio
async def test_rerun_with_same_checkpoint_does_not_resubmit(tmp_path):
    checkpoint = tmp_path / "bulk.json"
    store: t.Dict[str, t.Any] = {}
    prompts = ["a", "b"]

    first = FakeBulkLLM(BulkLLMLimits(), store=store, checkpoint_path=checkpoint)
    await first.generate_bulk(prompts)
    assert len(store["submitted"]) == 1

    second = FakeBulkLLM(BulkLLMLimits(), store=store, checkpoint_path=checkpoint)
    responses = await second.generate_bulk(prompts)

    assert responses == ["answer:a", "answer:b"]
    assert len(store["submitted"]) == 1


@pytest.mark.asyncio
async def test_changed_prompt_is_resent_on_resume(tmp_path):
    checkpoint = tmp_path / "bulk.json"
    store: t.Dict[str, t.Any] = {}

    first = FakeBulkLLM(BulkLLMLimits(), store=store, checkpoint_path=checkpoint)
    await first.generate_bulk(["old"], custom_ids=["0"])

    second = FakeBulkLLM(BulkLLMLimits(), store=store, checkpoint_path=checkpoint)
    responses = await second.generate_bulk(["new"], custom_ids=["0"])

    assert responses == ["answer:new"]
    assert len(store["submitted"]) == 2


@pytest.mark.asyncio
async def test_duplicate_custom_ids_are_rejected():
    llm = FakeBulkLLM(BulkLLMLimits(), store={})
    with pytest.raises(ValueError):
        await llm.generate_bulk(["a", "b"], custom_ids=["x", "x"])


def test_default_checkpoint_path_is_used_when_not_given():
    default = (
        inspect.signature(BaseRagasBulkLLM.__init__)
        .parameters["checkpoint_path"]
        .default
    )
    assert default == DEFAULT_CHECKPOINT_PATH
    assert DEFAULT_CHECKPOINT_PATH == Path(".ragas_bulk") / "checkpoint.json"


@pytest.mark.asyncio
async def test_rerun_retries_requests_that_failed_before(tmp_path):
    checkpoint = tmp_path / "bulk.json"
    first = FakeBulkLLM(
        BulkLLMLimits(),
        store={},
        checkpoint_path=checkpoint,
        fail_counts={"1": 100},
    )
    responses = await first.generate_bulk(
        ["a", "b"], custom_ids=["0", "1"], raise_exceptions=False
    )
    assert responses == ["answer:a", None]

    second_store: t.Dict[str, t.Any] = {}
    second = FakeBulkLLM(
        BulkLLMLimits(), store=second_store, checkpoint_path=checkpoint
    )
    responses = await second.generate_bulk(["a", "b"], custom_ids=["0", "1"])

    assert responses == ["answer:a", "answer:b"]
    assert second_store["submitted"] == [["1"]]


@pytest.mark.asyncio
async def test_usage_totals_are_logged_and_persisted(tmp_path, caplog):
    checkpoint = tmp_path / "bulk.json"
    llm = FakeBulkLLM(
        BulkLLMLimits(max_requests=2), store={}, checkpoint_path=checkpoint
    )

    with caplog.at_level(logging.INFO):
        await llm.generate_bulk(["a", "b", "c"], custom_ids=["0", "1", "2"])

    usage = llm.usage()
    assert usage.batches_submitted == 2
    assert usage.batches_completed == 2
    assert usage.requests_completed == 3
    assert usage.requests_failed == 0
    assert "Bulk LLM usage" in caplog.text
    assert BulkUsage.from_checkpoint(BulkCheckpoint.load(checkpoint)) == usage


class _EchoInput(BaseModel):
    text: str


class _EchoOutput(BaseModel):
    text: str


class _EchoPrompt(PydanticPrompt[_EchoInput, _EchoOutput]):
    name: str = "echo"
    instruction: str = "Echo the text as JSON."
    input_model = _EchoInput
    output_model = _EchoOutput


@pytest.mark.asyncio
async def test_prompt_generate_bulk_dedupes_and_parses():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(BulkLLMLimits(), store=store, respond=lambda _: '{"text": "out"}')
    inputs = [_EchoInput(text="x"), _EchoInput(text="x"), _EchoInput(text="y")]

    outputs = await _EchoPrompt().generate_bulk(llm, inputs)

    assert [o.text for o in outputs] == ["out", "out", "out"]
    assert sum(len(batch) for batch in store["submitted"]) == 2


@pytest.mark.asyncio
async def test_summary_extractor_bulk_path_sets_property():
    store: t.Dict[str, t.Any] = {}
    bulk = FakeBulkLLM(
        BulkLLMLimits(max_requests=10),
        store=store,
        respond=lambda _: '{"text": "short summary"}',
    )
    extractor = SummaryExtractor(llm=MagicMock(), bulk_llm=bulk)
    kg = KnowledgeGraph(
        nodes=[
            Node(properties={"page_content": "first doc"}),
            Node(properties={"page_content": "second doc"}),
        ]
    )

    results = await extractor.transform(kg)

    assert [node for node, _ in results] == kg.nodes
    assert [value for _, (_, value) in results] == ["short summary", "short summary"]
    assert len(store["submitted"]) == 1


def _flatten(transforms: t.Any) -> t.Iterator[t.Any]:
    for item in transforms:
        if isinstance(item, Parallel):
            yield from _flatten(item.transformations)
        else:
            yield item


class _CountingLLM(FakeBulkLLM):
    """Tracks how many batches are open (submitted and not yet finished)."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self.open_batches = 0
        self.max_open = 0

    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        batch_id = await super().submit_batch(requests)
        await asyncio.sleep(0)
        self.open_batches += 1
        self.max_open = max(self.max_open, self.open_batches)
        return batch_id

    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        result = await super().get_batch(batch_id)
        if result.status != "in_progress":
            self.open_batches -= 1
        return result


@pytest.mark.asyncio
async def test_in_flight_batches_are_capped():
    llm = _CountingLLM(
        BulkLLMLimits(max_requests=1),
        store={},
        in_progress_polls=4,
        max_in_flight_batches=2,
    )
    prompts = ["a", "b", "c", "d", "e"]

    responses = await llm.generate_bulk(prompts)

    assert responses == [f"answer:{p}" for p in prompts]
    assert llm.max_open == 2


def test_in_flight_cap_must_be_positive():
    with pytest.raises(ValueError):
        FakeBulkLLM(BulkLLMLimits(), store={}, max_in_flight_batches=0)
    with pytest.raises(ValueError):
        FakeBulkLLM(BulkLLMLimits(), store={}, max_batch_wait=0)
    with pytest.raises(ValueError):
        FakeBulkLLM(BulkLLMLimits(), store={}, max_transport_retries=-1)


class _FlakyLLM(FakeBulkLLM):
    """Raises a configured number of transport errors from submit or poll."""

    def __init__(
        self,
        *args: t.Any,
        submit_errors: t.Optional[t.List[BulkTransportError]] = None,
        poll_errors: t.Optional[t.List[BulkTransportError]] = None,
        **kwargs: t.Any,
    ):
        super().__init__(*args, **kwargs)
        self.transport_backoff = 0
        self.submit_errors = list(submit_errors or [])
        self.poll_errors = list(poll_errors or [])
        self.submit_calls = 0

    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        self.submit_calls += 1
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return await super().submit_batch(requests)

    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        if self.poll_errors:
            raise self.poll_errors.pop(0)
        return await super().get_batch(batch_id)


@pytest.mark.asyncio
async def test_retryable_submit_error_is_retried():
    llm = _FlakyLLM(
        BulkLLMLimits(),
        store={},
        submit_errors=[BulkTransportError("HTTP 429", retryable=True)],
    )

    responses = await llm.generate_bulk(["a"])

    assert responses == ["answer:a"]
    assert llm.submit_calls == 2


@pytest.mark.asyncio
async def test_non_retryable_submit_error_is_raised_immediately():
    llm = _FlakyLLM(
        BulkLLMLimits(),
        store={},
        submit_errors=[BulkTransportError("HTTP 401", retryable=False)],
    )

    with pytest.raises(BulkTransportError):
        await llm.generate_bulk(["a"])
    assert llm.submit_calls == 1


@pytest.mark.asyncio
async def test_transient_poll_failures_are_tolerated():
    llm = _FlakyLLM(
        BulkLLMLimits(),
        store={},
        poll_errors=[
            BulkTransportError("HTTP 503", retryable=True, status_code=503),
            BulkTransportError("connection reset", retryable=True),
        ],
    )

    responses = await llm.generate_bulk(["a"])

    assert responses == ["answer:a"]


@pytest.mark.asyncio
async def test_poll_gives_up_after_max_poll_failures():
    llm = _FlakyLLM(
        BulkLLMLimits(),
        store={},
        poll_errors=[BulkTransportError("HTTP 503", retryable=True)] * 5,
        max_transport_retries=0,
    )
    llm.max_poll_failures = 3

    with pytest.raises(BulkTransportError):
        await llm.generate_bulk(["a"])
    assert len(llm.poll_errors) == 2


class _StuckFirstBatchLLM(FakeBulkLLM):
    """The first batch never finishes; the fake clock advances on every poll."""

    def __init__(self, *args: t.Any, clock: t.Dict[str, float], **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self.clock = clock

    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        self.clock["now"] += 10
        if batch_id == "batch-1":
            return BulkBatchResult(status="in_progress")
        return await super().get_batch(batch_id)


@pytest.mark.asyncio
async def test_expired_batch_requests_are_retried_in_a_new_batch(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(bulk_module, "time", SimpleNamespace(time=lambda: clock["now"]))
    store: t.Dict[str, t.Any] = {}
    llm = _StuckFirstBatchLLM(
        BulkLLMLimits(), store=store, clock=clock, max_batch_wait=25
    )

    responses = await llm.generate_bulk(["a"])

    assert responses == ["answer:a"]
    assert len(store["submitted"]) == 2


@pytest.mark.asyncio
async def test_bulk_parse_failure_is_retried_through_bulk_llm():
    def respond(prompt: str) -> str:
        if "did not satisfy" in prompt:
            return '{"text": "{\\"text\\": \\"fixed\\"}"}'
        return "not json"

    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(BulkLLMLimits(), store=store, respond=respond)

    outputs = await _EchoPrompt().generate_bulk(llm, [_EchoInput(text="x")])

    assert [o.text for o in outputs] == ["fixed"]
    assert len(store["submitted"]) == 2
    assert store["submitted"][1][0].endswith(":fix1")


@pytest.mark.asyncio
async def test_bulk_parse_failure_raises_after_retries_are_exhausted():
    store: t.Dict[str, t.Any] = {}
    llm = FakeBulkLLM(BulkLLMLimits(), store=store, respond=lambda _: "not json")

    with pytest.raises(RagasOutputParserException):
        await _EchoPrompt().generate_bulk(llm, [_EchoInput(text="x")], retries_left=2)
    assert len(store["submitted"]) == 3


class _FakeHTTPResponse:
    def __init__(self, payload: t.Any):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: t.Any) -> bool:
        return False


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test",
        code,
        "error",
        email.message.Message(),
        io.BytesIO(body.encode("utf-8")),
    )


def _inline_llm(**kwargs: t.Any) -> InlineBatchBulkLLM:
    kwargs.setdefault("endpoint", "/v1/chat/completions")
    return InlineBatchBulkLLM(
        BulkLLMLimits(),
        model="vendor/model",
        base_url="https://example.test/api/v1/",
        api_key="secret-key",
        checkpoint_path=None,
        **kwargs,
    )


@pytest.fixture
def fake_urlopen(monkeypatch):
    """Replace urlopen with a queue of responses; records every request."""
    calls: t.List[t.Any] = []
    queue: t.List[t.Any] = []

    def fake(request, timeout=None):
        calls.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeHTTPResponse(item)

    monkeypatch.setattr(inline_batch_module, "_open", fake)
    return calls, queue


@pytest.mark.asyncio
async def test_inline_submit_puts_requests_last_and_sends_auth(fake_urlopen):
    calls, queue = fake_urlopen
    queue.append({"id": "b1", "status": "validating"})
    llm = _inline_llm(batch_options={"provider": {"order": ["x"]}})

    batch_id = await llm.submit_batch([_request("c1")])

    assert batch_id == "b1"
    request = calls[0]
    assert request.get_method() == "POST"
    assert request.full_url == "https://example.test/api/v1/batches"
    assert request.get_header("Authorization") == "Bearer secret-key"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "vendor/model"
    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["provider"] == {"order": ["x"]}
    assert list(payload)[-1] == "requests"


@pytest.mark.asyncio
async def test_inline_get_maps_results_and_usage(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(
        {
            "id": "b1",
            "status": "completed",
            "results": [
                {
                    "custom_id": "c1",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": "hi"}}]},
                    },
                    "error": None,
                },
                {
                    "custom_id": "c2",
                    "response": None,
                    "error": {"message": "content filtered"},
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.001},
        }
    )

    result = await _inline_llm().get_batch("b1")

    assert result.status == "completed"
    assert result.responses == {"c1": "hi"}
    assert result.errors == {"c2": "content filtered"}
    assert (result.input_tokens, result.output_tokens, result.cost) == (5, 2, 0.001)


@pytest.mark.asyncio
async def test_inline_get_reads_responses_endpoint_output(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(
        {
            "id": "b1",
            "status": "completed",
            "results": [
                {
                    "custom_id": "c1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {"type": "output_text", "text": "hel"},
                                        {"type": "output_text", "text": "lo"},
                                    ],
                                }
                            ],
                        },
                    },
                    "error": None,
                }
            ],
        }
    )

    result = await _inline_llm(endpoint="/v1/responses").get_batch("b1")

    assert result.responses == {"c1": "hello"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"id": "b1", "status": "in_progress"},
        {"id": "b1", "status": "finalizing"},
        {"id": "b1", "status": "completed", "results": None},
    ],
)
async def test_inline_get_reports_unfinished_batches_as_in_progress(
    fake_urlopen, payload
):
    _, queue = fake_urlopen
    queue.append(payload)

    result = await _inline_llm().get_batch("b1")

    assert result.status == "in_progress"


@pytest.mark.asyncio
async def test_inline_get_missing_batch_is_reported_as_failed(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(_http_error(404, '{"error": {"message": "not found"}}'))

    result = await _inline_llm().get_batch("b1")

    assert result.status == "failed"
    assert "404" in (result.error or "")


@pytest.mark.asyncio
async def test_inline_submit_rate_limit_is_retryable(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(_http_error(429))

    with pytest.raises(BulkTransportError) as info:
        await _inline_llm().submit_batch([_request("c1")])
    assert info.value.retryable is True
    assert info.value.status_code == 429


@pytest.mark.asyncio
async def test_inline_submit_server_error_is_not_retried(fake_urlopen):
    # A failed POST may still have created a batch, so it must not be retried.
    _, queue = fake_urlopen
    queue.append(_http_error(500))

    with pytest.raises(BulkTransportError) as info:
        await _inline_llm().submit_batch([_request("c1")])
    assert info.value.retryable is False


@pytest.mark.asyncio
async def test_inline_get_server_error_is_retryable(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(_http_error(503))

    with pytest.raises(BulkTransportError) as info:
        await _inline_llm().get_batch("b1")
    assert info.value.retryable is True


@pytest.mark.asyncio
async def test_inline_get_connection_error_is_retryable(fake_urlopen):
    _, queue = fake_urlopen
    queue.append(urllib.error.URLError("connection refused"))

    with pytest.raises(BulkTransportError) as info:
        await _inline_llm().get_batch("b1")
    assert info.value.retryable is True


def test_inline_rejects_unsupported_endpoint_and_requests_option():
    with pytest.raises(ValueError):
        _inline_llm(endpoint="/v1/embeddings")
    with pytest.raises(ValueError):
        _inline_llm(batch_options={"requests": []})


def test_default_prechunked_transforms_pass_bulk_llm_to_extractors():
    bulk = FakeBulkLLM(BulkLLMLimits(), store={})
    transforms = default_transforms_for_prechunked(
        llm=MagicMock(), embedding_model=MagicMock(), bulk_llm=bulk
    )
    extractors = [
        item
        for item in _flatten(transforms)
        if isinstance(item, (SummaryExtractor, ThemesExtractor, NERExtractor))
    ]

    assert {type(e) for e in extractors} == {
        SummaryExtractor,
        ThemesExtractor,
        NERExtractor,
    }
    assert all(e.bulk_llm is bulk for e in extractors)


def test_generate_with_chunks_rejects_transforms_with_bulk_llm():
    generator = TestsetGenerator(llm=MagicMock(), embedding_model=MagicMock())
    bulk = FakeBulkLLM(BulkLLMLimits(), store={})

    with pytest.raises(ValueError, match="transforms_bulk_llm"):
        generator.generate_with_chunks(
            ["chunk"], testset_size=1, transforms=[], transforms_bulk_llm=bulk
        )


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_checkpoint_and_one_cap(tmp_path):
    path = tmp_path / "bulk.json"
    llm = _CountingLLM(
        BulkLLMLimits(max_requests=1),
        store={},
        checkpoint_path=path,
        in_progress_polls=6,
        max_in_flight_batches=2,
    )
    groups = [[f"{g}{i}" for i in range(3)] for g in "abc"]

    results = await asyncio.gather(*[llm.generate_bulk(p, list(p)) for p in groups])

    assert results == [[f"answer:{x}" for x in p] for p in groups]
    assert llm.max_open == 2
    assert len(llm.store["submitted"]) == 9
    saved = BulkCheckpoint.load(path)
    assert len(saved.batches) == 9
    assert sorted(saved.requests) == sorted(x for p in groups for x in p)
    assert llm.usage().batches_completed == 9


@pytest.mark.asyncio
async def test_concurrent_calls_with_the_same_request_submit_it_once():
    llm = _CountingLLM(BulkLLMLimits(), store={}, in_progress_polls=2)

    first, second = await asyncio.gather(
        llm.generate_bulk(["a", "b"], ["x", "y"]),
        llm.generate_bulk(["a", "b"], ["x", "y"]),
    )

    assert first == second == ["answer:a", "answer:b"]
    submitted = [cid for batch in llm.store["submitted"] for cid in batch]
    assert sorted(submitted) == ["x", "y"]


def test_body_options_are_merged_into_every_request_body():
    options = {"temperature": 0.01, "max_tokens": 50}
    llm = FakeBulkLLM(BulkLLMLimits(), store={}, body_options=options)
    assert llm.build_body("hi") == {
        "messages": [{"role": "user", "content": "hi"}],
        **options,
    }
    inline = _inline_llm(endpoint="/v1/responses", body_options=options)
    assert inline.build_body("hi") == {"input": "hi", **options}


def test_body_options_cannot_replace_the_prompt():
    with pytest.raises(ValueError):
        FakeBulkLLM(BulkLLMLimits(), store={}, body_options={"messages": []})
    with pytest.raises(ValueError):
        FakeBulkLLM(BulkLLMLimits(), store={}, body_options={"input": "x"})


@pytest.mark.asyncio
async def test_checkpoint_from_other_settings_is_rejected(tmp_path):
    path = tmp_path / "bulk.json"
    options = {"temperature": 0}
    first = FakeBulkLLM(
        BulkLLMLimits(), store={}, checkpoint_path=path, body_options=options
    )
    await first.generate_bulk(["a"])

    store: t.Dict[str, t.Any] = {}
    same = FakeBulkLLM(
        BulkLLMLimits(), store=store, checkpoint_path=path, body_options=options
    )
    assert await same.generate_bulk(["a"]) == ["answer:a"]
    assert "submitted" not in store

    other = FakeBulkLLM(
        BulkLLMLimits(), store={}, checkpoint_path=path, body_options={"temperature": 1}
    )
    with pytest.raises(BulkCheckpointMismatchError):
        await other.generate_bulk(["a"])


def test_inline_fingerprint_covers_model_endpoint_and_options_but_not_the_key():
    base = _inline_llm()
    assert base.fingerprint() == _inline_llm().fingerprint()
    other_model = InlineBatchBulkLLM(
        BulkLLMLimits(),
        model="vendor/other",
        base_url="https://example.test/api/v1",
        api_key="secret-key",
        checkpoint_path=None,
    )
    assert base._fingerprint_digest() != other_model._fingerprint_digest()
    other_options = _inline_llm(batch_options={"provider": {"order": ["x"]}})
    assert base._fingerprint_digest() != other_options._fingerprint_digest()
    assert "secret-key" not in json.dumps(base.fingerprint())


@pytest.mark.asyncio
async def test_checkpoint_is_an_append_only_journal(tmp_path):
    path = tmp_path / "bulk.json"
    llm = FakeBulkLLM(BulkLLMLimits(max_requests=1), store={}, checkpoint_path=path)

    await llm.generate_bulk(["a", "b"])

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["version"] == 2
    assert header["fingerprint"]
    kinds = {json.loads(line)["t"] for line in lines[1:]}
    assert {"request", "batch", "intent", "settled"} <= kinds


def test_checkpoint_ignores_a_partial_last_line_and_rewrites_it(tmp_path):
    path = tmp_path / "bulk.json"
    checkpoint = BulkCheckpoint(path, "fp")
    checkpoint.requests["a"] = BulkRequestState(custom_id="a", prompt="p")
    checkpoint.touch_request("a")
    checkpoint.save()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"t": "request", "d": {"custom_id": "b"')

    loaded = BulkCheckpoint.load(path, "fp")
    assert list(loaded.requests) == ["a"]
    loaded.save()

    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_checkpoint_rejects_corrupt_and_foreign_files(tmp_path):
    path = tmp_path / "bulk.json"
    header = json.dumps({"version": 2, "fingerprint": None})
    path.write_text(f"{header}\nnot json\n{header}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        BulkCheckpoint.load(path)

    path.write_text('{"version": 1, "requests": {}, "batches": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        BulkCheckpoint.load(path)

    path.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a bulk LLM checkpoint"):
        BulkCheckpoint.load(path)


def test_checkpoint_journal_is_compacted_when_it_has_many_stale_records(tmp_path):
    path = tmp_path / "bulk.json"
    checkpoint = BulkCheckpoint(path, "fp")
    checkpoint.requests["a"] = BulkRequestState(custom_id="a", prompt="p")
    for _ in range(BulkCheckpoint.COMPACT_SLACK + 20):
        checkpoint.touch_request("a")
        checkpoint.save()

    loaded = BulkCheckpoint.load(path, "fp")
    loaded.save()

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.asyncio
async def test_interrupted_submit_is_reported_when_resuming(tmp_path, caplog):
    path = tmp_path / "bulk.json"
    crashing = FakeBulkLLM(
        BulkLLMLimits(), store={}, checkpoint_path=path, fail_submit_on=0
    )
    with pytest.raises(ConnectionError):
        await crashing.generate_bulk(["a"])

    resumed = FakeBulkLLM(BulkLLMLimits(), store={}, checkpoint_path=path)
    with caplog.at_level(logging.WARNING):
        assert await resumed.generate_bulk(["a"]) == ["answer:a"]

    assert "interrupted" in caplog.text
    assert resumed.usage().batches_submitted == 1


@pytest.mark.asyncio
async def test_rejected_submit_leaves_no_interrupted_record(tmp_path):
    path = tmp_path / "bulk.json"
    flaky = _FlakyLLM(
        BulkLLMLimits(),
        store={},
        checkpoint_path=path,
        submit_errors=[BulkTransportError("HTTP 429", retryable=True)],
        max_transport_retries=0,
    )
    with pytest.raises(BulkTransportError):
        await flaky.generate_bulk(["a"])

    assert BulkCheckpoint.load(path).interrupted_submits == []


@pytest.mark.asyncio
async def test_unparseable_answers_are_discarded_so_a_rerun_asks_again(tmp_path):
    path = tmp_path / "bulk.json"
    store: t.Dict[str, t.Any] = {}
    bad = FakeBulkLLM(
        BulkLLMLimits(), store=store, checkpoint_path=path, respond=lambda _: "not json"
    )
    with pytest.raises(RagasOutputParserException):
        await _EchoPrompt().generate_bulk(bad, [_EchoInput(text="x")], retries_left=1)
    assert len(store["submitted"]) == 2

    good = FakeBulkLLM(
        BulkLLMLimits(),
        store=store,
        checkpoint_path=path,
        respond=lambda _: '{"text": "ok"}',
    )
    outputs = await _EchoPrompt().generate_bulk(good, [_EchoInput(text="x")])

    assert [o.text for o in outputs] == ["ok"]
    assert len(store["submitted"]) == 3


@pytest.mark.asyncio
async def test_prompt_generate_bulk_can_return_none_for_unusable_items():
    llm = FakeBulkLLM(
        BulkLLMLimits(),
        store={},
        respond=lambda p: "not json" if "zzbad" in p else '{"text": "ok"}',
    )

    outputs = await _EchoPrompt().generate_bulk(
        llm,
        [_EchoInput(text="zzbad"), _EchoInput(text="fine")],
        retries_left=1,
        raise_exceptions=False,
    )

    assert [o.text if o else None for o in outputs] == [None, "ok"]


@pytest.mark.asyncio
async def test_extractor_can_keep_going_when_one_node_is_unusable():
    def respond(prompt: str) -> str:
        return "not json" if "zzbad" in prompt else '{"text": "ok"}'

    kg = KnowledgeGraph(
        nodes=[
            Node(properties={"page_content": "zzbad doc"}),
            Node(properties={"page_content": "fine doc"}),
        ]
    )
    tolerant = SummaryExtractor(
        llm=MagicMock(),
        bulk_llm=FakeBulkLLM(BulkLLMLimits(), store={}, respond=respond),
        bulk_raise_exceptions=False,
    )
    results = await tolerant.transform(kg)
    assert [value for _, (_, value) in results] == [None, "ok"]

    strict = SummaryExtractor(
        llm=MagicMock(),
        bulk_llm=FakeBulkLLM(BulkLLMLimits(), store={}, respond=respond),
    )
    with pytest.raises(RagasOutputParserException):
        await strict.transform(kg)


@pytest.mark.parametrize(
    "url", ["http://example.com/api", "ftp://example.com/api", "file:///tmp/x"]
)
def test_inline_requires_https_except_for_localhost(url):
    with pytest.raises(ValueError, match="https"):
        InlineBatchBulkLLM(
            BulkLLMLimits(), model="m", base_url=url, api_key="k", checkpoint_path=None
        )
    InlineBatchBulkLLM(
        BulkLLMLimits(),
        model="m",
        base_url="http://localhost:8080/v1",
        api_key="k",
        checkpoint_path=None,
    )


class _BatchHandler(http.server.BaseHTTPRequestHandler):
    def _reply(self, code: int, payload: t.Any = None, location: str = "") -> None:
        self.send_response(code)
        if location:
            self.send_header("Location", location)
        body = json.dumps(payload or {}).encode("utf-8")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        server: t.Any = self.server
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        server.seen.append((self.command, self.path, self.headers.get("Authorization")))
        server.batches["b1"] = [
            {
                "custom_id": r["custom_id"],
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": "echo:"
                                    + r["body"]["messages"][0]["content"]
                                }
                            }
                        ]
                    },
                },
                "error": None,
            }
            for r in payload["requests"]
        ]
        self._reply(202, {"id": "b1", "status": "validating"})

    def do_GET(self) -> None:
        server: t.Any = self.server
        server.seen.append((self.command, self.path, self.headers.get("Authorization")))
        if self.path.endswith("/redirect"):
            self._reply(
                302, location=f"http://127.0.0.1:{server.server_port}/elsewhere"
            )
        else:
            self._reply(
                200,
                {
                    "id": "b1",
                    "status": "completed",
                    "results": server.batches["b1"],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.5},
                },
            )

    def log_message(self, *args: t.Any) -> None:
        pass


@pytest.fixture
def batch_server():
    server: t.Any = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _BatchHandler)
    server.seen = []
    server.batches = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _local_llm(server: t.Any) -> InlineBatchBulkLLM:
    return InlineBatchBulkLLM(
        BulkLLMLimits(),
        model="m",
        base_url=f"http://127.0.0.1:{server.server_port}/api/v1",
        api_key="k",
        checkpoint_path=None,
        poll_interval=0,
    )


@pytest.mark.asyncio
async def test_inline_transport_round_trip_over_real_http(batch_server):
    llm = _local_llm(batch_server)

    responses = await llm.generate_bulk(["a", "b"])

    assert responses == ["echo:a", "echo:b"]
    assert batch_server.seen[0] == ("POST", "/api/v1/batches", "Bearer k")
    assert batch_server.seen[1] == ("GET", "/api/v1/batches/b1", "Bearer k")
    assert llm.usage().cost == 0.5


@pytest.mark.asyncio
async def test_inline_transport_does_not_follow_redirects(batch_server):
    llm = _local_llm(batch_server)

    with pytest.raises(BulkTransportError) as info:
        await llm.get_batch("redirect")

    assert info.value.status_code == 302
    assert info.value.retryable is False
    assert all(path != "/elsewhere" for _, path, _ in batch_server.seen)
