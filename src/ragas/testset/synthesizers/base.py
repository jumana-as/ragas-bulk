from __future__ import annotations

import asyncio
import math
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

from ragas.callbacks import new_group
from ragas.llms import BaseRagasLLM, llm_factory
from ragas.prompt import PromptMixin
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.dataset_schema import BaseSample
    from ragas.llms.base import InstructorBaseRagasLLM

_T = t.TypeVar("_T")

#: Used when an LLM has no run config, matching ``RunConfig.max_workers``.
DEFAULT_CONCURRENCY = 16


def wave_size(remaining: int, per_item: int) -> int:
    """
    How many items to process together to produce ``remaining`` results.

    Each item yields at most ``per_item`` results, so this many are needed anyway:
    running them together makes no extra calls compared with running them one by one.
    """
    return max(1, math.ceil(remaining / max(1, per_item)))


async def gather_limited(
    llm: t.Any, awaitables: t.Sequence[t.Awaitable[_T]]
) -> t.List[_T]:
    """
    Run independent LLM calls concurrently and return their results in order.

    At most ``llm.run_config.max_workers`` run at once (all of them when it is
    -1). The first failure cancels the rest and is raised.

    Calls that are awaited one after another cannot share a batch on a bulk LLM,
    so scenario generation issues the prompts of a wave through this helper.
    """
    if not awaitables:
        return []
    limit = getattr(
        getattr(llm, "run_config", None), "max_workers", DEFAULT_CONCURRENCY
    )
    semaphore = asyncio.Semaphore(limit) if limit and limit > 0 else None

    async def run(awaitable: t.Awaitable[_T]) -> _T:
        if semaphore is None:
            return await awaitable
        async with semaphore:
            return await awaitable

    tasks = [asyncio.ensure_future(run(a)) for a in awaitables]
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except asyncio.CancelledError:
        pending = {task for task in tasks if not task.done()}
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in tasks:
        if not task.cancelled() and task.exception() is not None:
            raise t.cast(BaseException, task.exception())
    return [task.result() for task in tasks]


def _default_llm_factory() -> t.Union[BaseRagasLLM, "InstructorBaseRagasLLM"]:
    """Create a default LLM instance with OpenAI gpt-4o-mini.

    Returns InstructorBaseRagasLLM instance which satisfies BaseRagasLLM interface.
    """
    from openai import OpenAI

    client = OpenAI()
    return llm_factory("gpt-4o-mini", client=client)


class QueryLength(str, Enum):
    """
    Enumeration of query lengths. Available options are: LONG, MEDIUM, SHORT
    """

    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


class QueryStyle(str, Enum):
    """
    Enumeration of query styles. Available options are: MISSPELLED, PERFECT_GRAMMAR, POOR_GRAMMAR, WEB_SEARCH_LIKE
    """

    MISSPELLED = "Misspelled queries"
    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class BaseScenario(BaseModel):
    """
    Base class for representing a scenario for generating test samples.

    Attributes
    ----------
    nodes : List[Node]
        List of nodes involved in the scenario.
    style : QueryStyle
        The style of the query.
    length : QueryLength
        The length of the query.
    persona : Persona
        A persona associated with the scenario.
    """

    nodes: t.List[Node]
    style: QueryStyle
    length: QueryLength
    persona: Persona


Scenario = t.TypeVar("Scenario", bound=BaseScenario)


@dataclass
class BaseSynthesizer(ABC, t.Generic[Scenario], PromptMixin):
    """
    Base class for synthesizing scenarios and samples.
    """

    name: str = ""
    llm: t.Union[BaseRagasLLM, "InstructorBaseRagasLLM"] = field(
        default_factory=_default_llm_factory
    )
    llm_context: t.Optional[str] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    async def generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: t.Optional[Callbacks] = None,
    ) -> t.List[Scenario]:
        callbacks = callbacks or []
        scenario_generation_rm, scenario_generation_group = new_group(
            name=self.name,
            inputs={"n": n, "knowledge_graph": str(knowledge_graph)},
            callbacks=callbacks,
        )
        scenarios = await self._generate_scenarios(
            n, knowledge_graph, persona_list, scenario_generation_group
        )
        scenario_generation_rm.on_chain_end(outputs={"scenarios": scenarios})
        return scenarios

    @abstractmethod
    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[Scenario]:
        pass

    async def generate_sample(
        self, scenario: Scenario, callbacks: t.Optional[Callbacks] = None
    ) -> BaseSample:
        callbacks = callbacks or []

        # new group for Sample Generation
        sample_generation_rm, sample_generation_grp = new_group(
            name=self.name,
            inputs={"scenario": scenario},
            callbacks=callbacks,
        )
        sample = await self._generate_sample(scenario, sample_generation_grp)
        sample_generation_rm.on_chain_end(outputs={"sample": sample})

        return sample

    @abstractmethod
    async def _generate_sample(
        self, scenario: Scenario, callbacks: Callbacks
    ) -> BaseSample:
        pass
