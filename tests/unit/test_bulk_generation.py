# The stub synthesizer is duck-typed, so these two checks do not apply to this file.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import asyncio
import random
import types
import typing as t
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.prompt_values import StringPromptValue

import ragas.testset.synthesizers.generate as generate_module
from ragas import SingleTurnSample
from ragas.llms import BulkLLMAdapter, BulkLLMLimits
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.generate import TestsetGenerator
from tests.unit.llms.test_bulk_adapter import RecordingBulkLLM

PERSONAS = [Persona(name="Researcher", role_description="Reads papers.")]


class StubSynthesizer:
    """One LLM call per scenario and one per sample, all made concurrently."""

    name = "stub"

    def __init__(self, llm: t.Any):
        self.llm = llm
        self.llm_seen_by_scenarios: t.List[t.Any] = []
        self.draws: t.List[t.Tuple[float, float]] = []

    async def generate_scenarios(
        self, n: int, knowledge_graph: t.Any, persona_list: t.Any, callbacks: t.Any
    ) -> t.List[t.Any]:
        self.llm_seen_by_scenarios.append(self.llm)
        self.draws.append((random.random(), float(np.random.random())))
        results = await asyncio.gather(
            *(
                self.llm.generate(StringPromptValue(text=f"scenario {i}"))
                for i in range(n)
            )
        )
        return [types.SimpleNamespace(topic=r.generations[0][0].text) for r in results]

    async def generate_sample(self, scenario: t.Any, callbacks: t.Any) -> t.Any:
        result = await self.llm.generate(
            StringPromptValue(text=f"sample for {scenario.topic}")
        )
        return SingleTurnSample(
            user_input=scenario.topic, reference=result.generations[0][0].text
        )


def _generator(llm: t.Any) -> TestsetGenerator:
    return TestsetGenerator(
        llm=llm,
        embedding_model=MagicMock(),
        knowledge_graph=KnowledgeGraph(),
        persona_list=list(PERSONAS),
    )


@pytest.fixture(autouse=True)
def _no_analytics(monkeypatch):
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")


def test_generate_sends_scenarios_and_samples_as_one_batch_each(fake_llm):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    synthesizer = StubSynthesizer(fake_llm)
    generator = _generator(fake_llm)

    testset = generator.generate(
        testset_size=6,
        query_distribution=[(synthesizer, 1.0)],
        bulk_llm=BulkLLMAdapter(bulk, collect_window=0.02),
    )

    assert len(testset.samples) == 6
    # Six scenario prompts went out together, then six sample prompts.
    assert [len(batch) for batch in bulk.store["submitted"]] == [6, 6]
    assert all(
        isinstance(llm, BulkLLMAdapter) for llm in synthesizer.llm_seen_by_scenarios
    )


def test_generate_puts_back_the_synthesizer_llm_and_run_config(fake_llm):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    synthesizer = StubSynthesizer(fake_llm)
    generator = _generator(fake_llm)
    previous_config = fake_llm.run_config

    generator.generate(
        testset_size=2,
        query_distribution=[(synthesizer, 1.0)],
        run_config=RunConfig(max_workers=4),
        bulk_llm=BulkLLMAdapter(bulk, collect_window=0.02),
    )

    assert synthesizer.llm is fake_llm
    assert fake_llm.run_config is previous_config


def test_generate_puts_back_the_synthesizer_llm_when_it_fails(fake_llm):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={}, fail_submit_on=0)
    synthesizer = StubSynthesizer(fake_llm)
    generator = _generator(fake_llm)

    with pytest.raises(ConnectionError):
        generator.generate(
            testset_size=2,
            query_distribution=[(synthesizer, 1.0)],
            bulk_llm=BulkLLMAdapter(bulk, collect_window=0.02),
        )

    assert synthesizer.llm is fake_llm


def test_generate_seeds_the_random_generators_from_the_run_config(fake_llm):
    random.seed(123)
    expected = (random.random(), float(np.random.RandomState(123).random_sample()))
    draws = []
    for _ in range(2):
        bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
        synthesizer = StubSynthesizer(fake_llm)
        _generator(fake_llm).generate(
            testset_size=2,
            query_distribution=[(synthesizer, 1.0)],
            run_config=RunConfig(seed=123),
            bulk_llm=BulkLLMAdapter(bulk, collect_window=0.02),
        )
        draws.append(synthesizer.draws[0])

    assert draws[0] == draws[1] == expected


def test_generate_does_not_touch_the_random_state_without_a_bulk_llm(fake_llm):
    synthesizer = StubSynthesizer(fake_llm)
    random.seed(5)
    expected = random.random()
    random.seed(5)

    _generator(fake_llm).generate(
        testset_size=1, query_distribution=[(synthesizer, 1.0)]
    )

    assert synthesizer.draws[0][0] == expected


def test_generate_rejects_return_executor_with_a_bulk_llm(fake_llm):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})

    with pytest.raises(ValueError, match="return_executor"):
        _generator(fake_llm).generate(
            testset_size=1,
            query_distribution=[(StubSynthesizer(fake_llm), 1.0)],
            return_executor=True,
            bulk_llm=bulk,
        )


def test_transforms_llm_cannot_be_combined_with_bulk_llm(fake_llm):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})

    with pytest.raises(ValueError, match="transforms_llm"):
        _generator(fake_llm).generate_with_chunks(
            ["chunk"], testset_size=1, transforms_llm=MagicMock(), bulk_llm=bulk
        )


@pytest.mark.parametrize("wrap", [False, True])
def test_bulk_llm_reaches_the_default_transforms_and_the_generate_stage(
    fake_llm, monkeypatch, wrap
):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    adapter = BulkLLMAdapter(bulk, collect_window=0.02)
    seen: t.Dict[str, t.Any] = {}

    def fake_default_transforms(**kwargs: t.Any) -> t.List[t.Any]:
        seen["transforms"] = kwargs
        return []

    def fake_apply_transforms(kg: t.Any, transforms: t.Any, run_config: t.Any = None):
        seen["run_config"] = run_config

    generator = _generator(fake_llm)
    monkeypatch.setattr(
        generate_module, "default_transforms_for_prechunked", fake_default_transforms
    )
    monkeypatch.setattr(generate_module, "apply_transforms", fake_apply_transforms)
    monkeypatch.setattr(
        generator, "generate", lambda **kwargs: seen.update(generate=kwargs)
    )

    generator.generate_with_chunks(
        ["chunk"],
        testset_size=1,
        run_config=RunConfig(max_workers=8),
        bulk_llm=adapter if wrap else bulk,
    )

    assert isinstance(seen["transforms"]["llm"], BulkLLMAdapter)
    if wrap:
        assert seen["transforms"]["llm"] is adapter
    assert seen["transforms"]["bulk_llm"] is bulk
    assert seen["run_config"].max_workers == -1
    assert seen["generate"]["bulk_llm"] is (adapter if wrap else bulk)


def test_transforms_bulk_llm_still_wins_for_the_extractors(fake_llm, monkeypatch):
    bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    extraction_bulk = RecordingBulkLLM(BulkLLMLimits(), store={})
    seen: t.Dict[str, t.Any] = {}
    generator = _generator(fake_llm)
    monkeypatch.setattr(
        generate_module,
        "default_transforms_for_prechunked",
        lambda **kwargs: seen.update(transforms=kwargs) or [],
    )
    monkeypatch.setattr(generate_module, "apply_transforms", lambda *a, **k: None)
    monkeypatch.setattr(generator, "generate", lambda **kwargs: None)

    generator.generate_with_chunks(
        ["chunk"],
        testset_size=1,
        transforms_bulk_llm=extraction_bulk,
        bulk_llm=bulk,
    )

    assert seen["transforms"]["bulk_llm"] is extraction_bulk
