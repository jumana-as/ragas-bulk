from __future__ import annotations

import asyncio
import typing as t

import pytest
from langchain_core.prompt_values import StringPromptValue
from pydantic import BaseModel

from ragas.llms import BulkGenerationError, BulkLLMAdapter, BulkLLMLimits
from ragas.llms.bulk import BulkRequest
from ragas.llms.bulk_adapter import _request_id
from ragas.prompt import PydanticPrompt
from tests.unit.llms.test_bulk import FakeBulkLLM


class RecordingBulkLLM(FakeBulkLLM):
    """FakeBulkLLM that also keeps the body of every submitted request."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self.bodies: t.Dict[str, t.Dict[str, t.Any]] = {}

    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        for request in requests:
            self.bodies[request.custom_id] = request.body
        return await super().submit_batch(requests)


def _prompt(text: str) -> StringPromptValue:
    return StringPromptValue(text=text)


def _texts(result: t.Any) -> t.List[str]:
    return [generation.text for generation in result.generations[0]]


def _adapter(bulk: RecordingBulkLLM, **kwargs: t.Any) -> BulkLLMAdapter:
    kwargs.setdefault("collect_window", 0.02)
    return BulkLLMAdapter(bulk, **kwargs)


@pytest.mark.asyncio
async def test_concurrent_calls_share_one_batch():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    results = await asyncio.gather(
        *(adapter.generate(_prompt(f"p{i}")) for i in range(5))
    )

    assert [_texts(r) for r in results] == [[f"answer:p{i}"] for i in range(5)]
    assert len(bulk.store["submitted"]) == 1
    assert len(bulk.store["submitted"][0]) == 5


@pytest.mark.asyncio
async def test_calls_after_the_window_go_in_a_later_batch():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk, collect_window=0.1)

    first = asyncio.create_task(adapter.generate(_prompt("a")))
    await asyncio.sleep(0.5)
    second = await adapter.generate(_prompt("b"))

    assert _texts(await first) == ["answer:a"]
    assert _texts(second) == ["answer:b"]
    assert [len(batch) for batch in bulk.store["submitted"]] == [1, 1]


@pytest.mark.asyncio
async def test_temperature_and_stop_are_sent_per_request():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    await asyncio.gather(
        adapter.generate(_prompt("a")),
        adapter.generate(_prompt("b"), temperature=1.0, stop=["END"]),
    )

    by_prompt = {body["messages"][0]["content"]: body for body in bulk.bodies.values()}
    assert by_prompt["a"]["temperature"] == 0.01
    assert "stop" not in by_prompt["a"]
    assert by_prompt["b"]["temperature"] == 1.0
    assert by_prompt["b"]["stop"] == ["END"]


@pytest.mark.asyncio
async def test_default_temperature_depends_on_n():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    await adapter.generate(_prompt("a"), temperature=None, n=2)

    assert {body["temperature"] for body in bulk.bodies.values()} == {0.3}


@pytest.mark.asyncio
async def test_temperature_can_be_left_out():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk, send_temperature=False)

    await adapter.generate(_prompt("a"))

    assert all("temperature" not in body for body in bulk.bodies.values())


@pytest.mark.asyncio
async def test_same_prompt_with_other_temperature_is_a_separate_request():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    await asyncio.gather(
        adapter.generate(_prompt("a"), temperature=0.1),
        adapter.generate(_prompt("a"), temperature=0.9),
    )

    assert len(bulk.store["submitted"][0]) == 2


@pytest.mark.asyncio
async def test_identical_concurrent_calls_share_one_request():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    results = await asyncio.gather(*(adapter.generate(_prompt("a")) for _ in range(3)))

    assert [_texts(r) for r in results] == [["answer:a"]] * 3
    assert sum(len(batch) for batch in bulk.store["submitted"]) == 1


@pytest.mark.asyncio
async def test_n_greater_than_one_sends_n_requests():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    result = await adapter.generate(_prompt("a"), n=3)

    assert _texts(result) == ["answer:a"] * 3
    assert len(bulk.store["submitted"][0]) == 3


@pytest.mark.asyncio
async def test_a_failed_request_fails_only_its_own_call_and_is_not_retried_again():
    bad_id = _request_id("bad", {"temperature": 0.01}, None)
    bulk = RecordingBulkLLM(
        BulkLLMLimits(),
        store={},
        fail_counts={bad_id: 100},
        max_attempts=1,
    )
    adapter = _adapter(bulk)

    good, bad = await asyncio.gather(
        adapter.generate(_prompt("good")),
        adapter.generate(_prompt("bad")),
        return_exceptions=True,
    )

    assert _texts(good) == ["answer:good"]
    assert isinstance(bad, BulkGenerationError)
    # A tenacity retry around the call would have submitted the request again.
    assert len(bulk.store["submitted"]) == 1


@pytest.mark.asyncio
async def test_a_transport_error_fails_every_call_in_the_flush():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={}, fail_submit_on=0)
    adapter = _adapter(bulk)

    results = await asyncio.gather(
        adapter.generate(_prompt("a")),
        adapter.generate(_prompt("b")),
        return_exceptions=True,
    )

    assert all(isinstance(r, ConnectionError) for r in results)


def test_the_adapter_works_on_successive_event_loops():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = _adapter(bulk)

    async def run(text: str) -> str:
        return _texts(await adapter.generate(_prompt(text)))[0]

    assert asyncio.run(run("a")) == "answer:a"
    assert asyncio.run(run("b")) == "answer:b"


def test_the_adapter_is_async_only_and_unbounded_by_default():
    adapter = _adapter(RecordingBulkLLM(BulkLLMLimits(), store={}))

    assert adapter.run_config.max_workers == -1
    with pytest.raises(NotImplementedError):
        adapter.generate_text(_prompt("a"))


def test_the_adapter_rejects_bad_timing_settings():
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    with pytest.raises(ValueError):
        BulkLLMAdapter(bulk, collect_window=-1)
    with pytest.raises(ValueError):
        BulkLLMAdapter(bulk, max_collect_wait=0)


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
async def test_prompts_run_unchanged_through_the_adapter():
    bulk = RecordingBulkLLM(
        BulkLLMLimits(), store={}, respond=lambda _: '{"text": "out"}'
    )
    adapter = _adapter(bulk)

    outputs = await asyncio.gather(
        *(
            _EchoPrompt().generate(adapter, data=_EchoInput(text=f"x{i}"))
            for i in range(4)
        )
    )

    assert [o.text for o in outputs] == ["out"] * 4
    assert len(bulk.store["submitted"]) == 1


@pytest.mark.asyncio
async def test_unparseable_output_is_fixed_through_a_second_batch():
    def respond(prompt: str) -> str:
        if "did not satisfy" in prompt:
            return '{"text": "{\\"text\\": \\"fixed\\"}"}'
        return "not json"

    bulk = RecordingBulkLLM(BulkLLMLimits(), store={}, respond=respond)
    adapter = _adapter(bulk)

    output = await _EchoPrompt().generate(adapter, data=_EchoInput(text="x"))

    assert output.text == "fixed"
    assert len(bulk.store["submitted"]) == 2
