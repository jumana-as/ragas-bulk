from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass

import numpy as np

from ragas.prompt import PydanticPrompt
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import gather_limited, wave_size
from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    ConceptCombinationPrompt,
    ConceptsList,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


@dataclass
class MultiHopAbstractQuerySynthesizer(MultiHopQuerySynthesizer):
    """Synthesize abstract multi-hop queries from given knowledge graph."""

    name: str = "multi_hop_abstract_query_synthesizer"
    relation_property: str = "summary_similarity"
    abstract_property_name: str = "themes"
    concept_combination_prompt: PydanticPrompt = ConceptCombinationPrompt()
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()

    def get_node_clusters(
        self,
        knowledge_graph: KnowledgeGraph,
        n: int = 1,
    ) -> t.List[t.Set[Node]]:
        """Find n indirect clusters of nodes based on relationship condition"""

        node_clusters = knowledge_graph.find_n_indirect_clusters(
            n,
            relationship_condition=lambda rel: (
                True if rel.get_property(self.relation_property) else False
            ),
            depth_limit=3,
        )
        logger.info("found %d clusters", len(node_clusters))
        return node_clusters

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[MultiHopScenario]:
        """
        Generate a list of scenarios of type MultiHopScenario.

        Steps to generate scenarios:
        1. Find n indirect clusters of nodes based on relationship condition
        2. Calculate the number of samples that should be created per cluster to get n samples in total
        3. For each cluster of nodes
            a. Find the child nodes of the cluster nodes
            b. Find list of personas that can be associated with the entities to create query
            c. Create all possible combinations of (nodes, entities, personas, style, length) as scenarios
        4. Sample diverse combinations of scenarios to get n samples
        """

        node_clusters = self.get_node_clusters(knowledge_graph, n)
        scenarios = []

        if len(node_clusters) == 0:
            raise ValueError(
                "No clusters found in the knowledge graph. Try changing the relationship condition."
            )
        num_sample_per_cluster = int(np.ceil(n / len(node_clusters)))

        child_relationships = [
            rel for rel in knowledge_graph.relationships if rel.type == "child"
        ]

        # Each cluster adds at most num_sample_per_cluster scenarios, so a wave only
        # takes the clusters still needed. The two prompts of a cluster depend on
        # each other, but the clusters of a wave do not, so each step runs together.
        position = 0
        while len(scenarios) < n and position < len(node_clusters):
            size = wave_size(n - len(scenarios), num_sample_per_cluster)
            wave = []
            for cluster in node_clusters[position : position + size]:
                nodes = []
                for node in cluster:
                    child_nodes = [
                        rel.target for rel in child_relationships if rel.source == node
                    ]
                    if child_nodes:
                        nodes.extend(child_nodes)
                    else:
                        nodes.append(node)
                wave.append(nodes)
            position += size

            combinations = await gather_limited(
                self.llm,
                [
                    self.concept_combination_prompt.generate(
                        data=ConceptsList(
                            lists_of_concepts=[
                                node.properties.get(self.abstract_property_name, [])
                                for node in nodes
                            ],
                            max_combinations=num_sample_per_cluster,
                        ),
                        llm=self.llm,
                        callbacks=callbacks,
                    )
                    for nodes in wave
                ],
            )
            matches = await gather_limited(
                self.llm,
                [
                    self.theme_persona_matching_prompt.generate(
                        data=ThemesPersonasInput(
                            themes=[
                                theme
                                for sublist in combination.combinations
                                for theme in sublist
                            ],
                            personas=persona_list,
                        ),
                        llm=self.llm,
                        callbacks=callbacks,
                    )
                    for combination in combinations
                ],
            )
            for nodes, combination, persona_concepts in zip(
                wave, combinations, matches
            ):
                if len(scenarios) >= n:
                    break
                base_scenarios = self.prepare_combinations(
                    nodes,
                    combination.combinations,
                    personas=persona_list,
                    persona_item_mapping=persona_concepts.mapping,
                    property_name=self.abstract_property_name,
                )
                base_scenarios = self.sample_diverse_combinations(
                    base_scenarios, num_sample_per_cluster
                )
                scenarios.extend(base_scenarios)

        return scenarios
