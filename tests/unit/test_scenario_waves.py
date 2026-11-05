from __future__ import annotations

import asyncio
import math
import random
import typing as t

import pytest

from ragas.prompt import PydanticPrompt
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import gather_limited, wave_size
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinations,
    ConceptsList,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.prompts import PersonaThemesMapping, ThemesPersonasInput
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from tests.unit.test_knowledge_graph_clusters import (
    build_knowledge_graph,
    create_chain_of_similarities,
    create_document_and_child_nodes,
)

PERSONAS = [
    Persona(name="Researcher", role_description="Reads papers."),
    Persona(name="Engineer", role_description="Builds systems."),
]


class Tracker:
    """Counts calls and how many run at the same time."""

    def __init__(self) -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def run(self) -> None:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1


class TrackedMatchingPrompt(PydanticPrompt):
    tracker: Tracker

    async def generate(self, data: ThemesPersonasInput, llm, callbacks=None):
        await self.tracker.run()
        return PersonaThemesMapping(
            mapping={persona.name: data.themes for persona in data.personas}
        )


class TrackedCombinationPrompt(PydanticPrompt):
    tracker: Tracker

    async def generate(self, data: ConceptsList, llm, callbacks=None):
        await self.tracker.run()
        return ConceptCombinations(
            combinations=data.lists_of_concepts[: data.max_combinations]
        )


def _matching(tracker: Tracker) -> TrackedMatchingPrompt:
    prompt = TrackedMatchingPrompt()
    prompt.tracker = tracker
    return prompt


def _combination(tracker: Tracker) -> TrackedCombinationPrompt:
    prompt = TrackedCombinationPrompt()
    prompt.tracker = tracker
    return prompt


def _chunk_graph(entities: t.List[t.List[str]]) -> KnowledgeGraph:
    nodes = []
    for index, names in enumerate(entities):
        node = Node(type=NodeType.CHUNK)
        node.add_property("page_content", f"chunk {index}")
        node.add_property("entities", names)
        nodes.append(node)
    return KnowledgeGraph(nodes=nodes)


def _describe(scenarios: t.List[t.Any]) -> t.List[t.Tuple]:
    return [
        (
            s.term,
            [str(node.id) for node in s.nodes],
            s.persona.name,
            s.style,
            s.length,
        )
        for s in scenarios
    ]


async def _sequential_single_hop(
    synthesizer: SingleHopSpecificQuerySynthesizer,
    n: int,
    kg: KnowledgeGraph,
) -> t.List[t.Any]:
    """The scenario loop as it was before waves, one prompt after another."""
    nodes = synthesizer.get_node_clusters(kg)
    samples_per_node = int(math.ceil(n / len(nodes)))
    scenarios: t.List[t.Any] = []
    for node in nodes:
        if len(scenarios) >= n:
            break
        themes = synthesizer._extract_themes_from_items(
            node.properties.get(synthesizer.property_name, [])
        )
        if not themes:
            continue
        concepts = await synthesizer.theme_persona_matching_prompt.generate(
            data=ThemesPersonasInput(themes=themes, personas=PERSONAS),
            llm=synthesizer.llm,
            callbacks=None,
        )
        base = synthesizer.prepare_combinations(
            node, themes, personas=PERSONAS, persona_concepts=concepts.mapping
        )
        scenarios.extend(synthesizer.sample_combinations(base, samples_per_node))
    return scenarios


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 3, 5, 8, 12])
async def test_single_hop_waves_give_the_same_scenarios_as_one_by_one(fake_llm, n):
    kg = _chunk_graph(
        [["a1", "a2"], [], ["b1"], ["c1", "c2", "c3"], ["d1"], [], ["e1", "e2"]]
    )
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(Tracker())

    random.seed(7)
    expected = await _sequential_single_hop(synthesizer, n, kg)
    random.seed(7)
    actual = await synthesizer._generate_scenarios(
        n=n, knowledge_graph=kg, persona_list=PERSONAS, callbacks=None
    )

    assert _describe(actual) == _describe(expected)


@pytest.mark.asyncio
async def test_single_hop_wave_asks_only_for_the_nodes_it_needs(fake_llm):
    tracker = Tracker()
    kg = _chunk_graph([[f"e{i}"] for i in range(10)])
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(tracker)

    scenarios = await synthesizer._generate_scenarios(
        n=4, knowledge_graph=kg, persona_list=PERSONAS, callbacks=None
    )

    assert len(scenarios) == 4
    assert tracker.calls == 4
    assert tracker.max_in_flight == 4


@pytest.mark.asyncio
async def test_single_hop_wave_honours_max_workers(fake_llm):
    tracker = Tracker()
    fake_llm.run_config = RunConfig(max_workers=2)
    kg = _chunk_graph([[f"e{i}"] for i in range(10)])
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(tracker)

    await synthesizer._generate_scenarios(
        n=8, knowledge_graph=kg, persona_list=PERSONAS, callbacks=None
    )

    assert tracker.calls == 8
    assert tracker.max_in_flight == 2


@pytest.mark.asyncio
async def test_single_hop_wave_is_unbounded_when_max_workers_is_unlimited(fake_llm):
    tracker = Tracker()
    fake_llm.run_config = RunConfig(max_workers=-1)
    kg = _chunk_graph([[f"e{i}"] for i in range(30)])
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(tracker)

    await synthesizer._generate_scenarios(
        n=30, knowledge_graph=kg, persona_list=PERSONAS, callbacks=None
    )

    assert tracker.max_in_flight == 30


@pytest.mark.asyncio
async def test_single_hop_wave_keeps_going_when_nodes_give_too_few_scenarios(fake_llm):
    tracker = Tracker()
    # Nodes without themes give nothing, so one wave is not enough for n scenarios.
    kg = _chunk_graph([[], [], [], [], ["x"], ["y"], ["z"], ["w"]])
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(tracker)

    scenarios = await synthesizer._generate_scenarios(
        n=4, knowledge_graph=kg, persona_list=PERSONAS, callbacks=None
    )

    assert len(scenarios) == 4
    assert tracker.calls == 4


def _overlap_graph(pairs: int) -> KnowledgeGraph:
    nodes = []
    relationships = []
    for index in range(pairs):
        source = Node(type=NodeType.CHUNK)
        target = Node(type=NodeType.CHUNK)
        for node, name in ((source, f"s{index}"), (target, f"t{index}")):
            node.add_property("page_content", name)
            node.add_property("entities", [name, f"{name}_x"])
        nodes.extend([source, target])
        relationships.append(
            Relationship(
                source=source,
                target=target,
                type="entities_overlap",
                properties={"overlapped_items": [(f"o{index}", f"o{index}_b")]},
            )
        )
    return KnowledgeGraph(nodes=nodes, relationships=relationships)


@pytest.mark.asyncio
async def test_multi_hop_specific_runs_the_needed_clusters_together(fake_llm):
    tracker = Tracker()
    synthesizer = MultiHopSpecificQuerySynthesizer(llm=fake_llm)
    synthesizer.theme_persona_matching_prompt = _matching(tracker)

    scenarios = await synthesizer._generate_scenarios(
        n=4,
        knowledge_graph=_overlap_graph(6),
        persona_list=PERSONAS,
        callbacks=None,
    )

    assert len(scenarios) == 4
    assert tracker.calls == 4
    assert tracker.max_in_flight == 4


@pytest.mark.asyncio
async def test_multi_hop_abstract_runs_each_step_for_all_clusters_together(fake_llm):
    nodes, relationships = create_document_and_child_nodes()
    sim_nodes, sim_relationships = create_chain_of_similarities(nodes[0], node_count=3)
    branch_nodes, branch_relationships = create_chain_of_similarities(
        sim_nodes[1], node_count=4
    )
    nodes.extend(sim_nodes[1:])
    nodes.extend(branch_nodes[1:])
    relationships.extend(sim_relationships)
    relationships.extend(branch_relationships)
    kg = build_knowledge_graph(nodes, relationships)

    combination_tracker = Tracker()
    matching_tracker = Tracker()
    synthesizer = MultiHopAbstractQuerySynthesizer(llm=fake_llm)
    synthesizer.concept_combination_prompt = _combination(combination_tracker)
    synthesizer.theme_persona_matching_prompt = _matching(matching_tracker)

    scenarios = await synthesizer._generate_scenarios(
        n=len(kg.nodes),
        knowledge_graph=kg,
        persona_list=PERSONAS,
        callbacks=None,
    )

    assert len(scenarios) >= len(kg.nodes)
    assert combination_tracker.max_in_flight > 1
    assert matching_tracker.max_in_flight > 1
    assert combination_tracker.calls == matching_tracker.calls


@pytest.mark.parametrize(
    ("remaining", "per_item", "expected"),
    [(4, 1, 4), (10, 2, 5), (5, 2, 3), (1, 10, 1), (0, 3, 1), (3, 0, 3)],
)
def test_wave_size(remaining, per_item, expected):
    assert wave_size(remaining, per_item) == expected


class _Llm:
    def __init__(self, max_workers: t.Optional[int]):
        if max_workers is not None:
            self.run_config = RunConfig(max_workers=max_workers)


@pytest.mark.asyncio
async def test_gather_limited_keeps_the_order_of_the_inputs():
    async def work(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    results = await gather_limited(
        _Llm(-1), [work(1, 0.03), work(2, 0.0), work(3, 0.01)]
    )

    assert results == [1, 2, 3]


@pytest.mark.asyncio
async def test_gather_limited_caps_concurrency_and_defaults_to_16():
    for max_workers, expected in ((3, 3), (None, 16)):
        tracker = Tracker()
        await gather_limited(_Llm(max_workers), [tracker.run() for _ in range(40)])
        assert tracker.max_in_flight == expected


@pytest.mark.asyncio
async def test_gather_limited_stops_the_rest_at_the_first_failure():
    finished: t.List[int] = []

    async def work(index: int) -> None:
        if index == 0:
            raise RuntimeError("boom")
        await asyncio.sleep(0.2)
        finished.append(index)

    with pytest.raises(RuntimeError, match="boom"):
        await gather_limited(_Llm(-1), [work(i) for i in range(5)])

    await asyncio.sleep(0.3)
    assert finished == []


@pytest.mark.asyncio
async def test_gather_limited_with_nothing_to_run():
    assert await gather_limited(_Llm(2), []) == []
