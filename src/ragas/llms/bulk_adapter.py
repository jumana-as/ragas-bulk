"""
Use a bulk (batch endpoint) LLM wherever a regular LLM is expected.

Code such as the testset synthesizers calls an LLM one prompt at a time, from many
concurrent tasks. :class:`BulkLLMAdapter` collects the prompts that arrive close
together and sends them to the bulk LLM as one call, so they are packed into
batches. Each caller still gets its own answer back, so prompts, output parsing
and the fix-output-format retries work unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import typing as t
import weakref

from langchain_core.outputs import Generation, LLMResult

from ragas.llms.base import BaseRagasLLM
from ragas.llms.bulk import BaseRagasBulkLLM, BulkGenerationError
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.prompt_values import PromptValue

logger = logging.getLogger(__name__)


def _request_id(
    prompt: str, options: t.Dict[str, t.Any], index: t.Optional[int]
) -> str:
    """Stable id of one request, so a rerun finds its answer in the checkpoint."""
    text = json.dumps([prompt, options, index], sort_keys=True, ensure_ascii=False)
    return "llm:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Collector:
    """Prompts waiting to be sent, for the calls made on one event loop."""

    def __init__(self, adapter: BulkLLMAdapter):
        self.adapter = adapter
        self.futures: t.Dict[str, asyncio.Future[str]] = {}
        self.queue: t.Dict[str, t.Tuple[str, t.Dict[str, t.Any]]] = {}
        self.first_arrival: t.Optional[float] = None
        self.last_arrival = 0.0
        self.flusher: t.Optional[asyncio.Task[None]] = None
        self.runs: t.Set[asyncio.Task[None]] = set()

    async def request(
        self, prompt: str, options: t.Dict[str, t.Any], count: int
    ) -> t.List[str]:
        loop = asyncio.get_running_loop()
        futures: t.List[asyncio.Future[str]] = []
        for index in range(count):
            cid = _request_id(prompt, options, index if count > 1 else None)
            future = self.futures.get(cid)
            if future is None:
                future = loop.create_future()
                self.futures[cid] = future
                self.queue[cid] = (prompt, options)
            futures.append(future)

        now = loop.time()
        if self.first_arrival is None:
            self.first_arrival = now
        self.last_arrival = now
        if self.flusher is None or self.flusher.done():
            self.flusher = loop.create_task(self._flush_loop())

        # Shielded, so cancelling one caller does not cancel a shared request.
        return list(await asyncio.gather(*(asyncio.shield(f) for f in futures)))

    async def _flush_loop(self) -> None:
        loop = asyncio.get_running_loop()
        adapter = self.adapter
        while self.queue:
            while True:
                now = loop.time()
                quiet = now - self.last_arrival
                waited = now - (self.first_arrival or now)
                if (
                    quiet >= adapter.collect_window
                    or waited >= adapter.max_collect_wait
                ):
                    break
                await asyncio.sleep(
                    min(
                        adapter.collect_window - quiet,
                        adapter.max_collect_wait - waited,
                    )
                )
            taken, self.queue = self.queue, {}
            self.first_arrival = None
            run = loop.create_task(self._run(taken))
            self.runs.add(run)
            run.add_done_callback(self.runs.discard)

    async def _run(self, taken: t.Dict[str, t.Tuple[str, t.Dict[str, t.Any]]]) -> None:
        ids = list(taken)
        bulk_llm = self.adapter.bulk_llm
        logger.info("Sending %d collected prompt(s) to the bulk LLM", len(ids))
        try:
            results = await bulk_llm.generate_bulk(
                [taken[cid][0] for cid in ids],
                custom_ids=ids,
                raise_exceptions=False,
                request_options=[taken[cid][1] for cid in ids],
            )
        except asyncio.CancelledError:
            for cid in ids:
                self._settle(cid, cancel=True)
            raise
        except Exception as exc:
            for cid in ids:
                self._settle(cid, error=exc)
            return
        for cid, text in zip(ids, results):
            if text is None:
                reason = bulk_llm.request_error(cid) or "no response"
                self._settle(cid, error=BulkGenerationError({cid: reason}))
            else:
                self._settle(cid, text=text)

    def _settle(
        self,
        cid: str,
        text: t.Optional[str] = None,
        error: t.Optional[BaseException] = None,
        cancel: bool = False,
    ) -> None:
        future = self.futures.pop(cid, None)
        if future is None or future.done():
            return
        if cancel:
            future.cancel()
        elif error is not None:
            future.set_exception(error)
            future.exception()  # mark as retrieved if every caller gave up
        else:
            future.set_result(t.cast(str, text))


class BulkLLMAdapter(BaseRagasLLM):
    """
    A regular LLM that answers through a bulk (batch endpoint) LLM.

    Calls made at about the same time are collected and sent together, so a
    pipeline of concurrent tasks turns into a few large batches. A call whose
    request still has no response after all attempts raises
    :class:`BulkGenerationError` for that call only.

    Collection only helps when calls are made concurrently. A loop that awaits one
    call after another sends one batch per call.

    Parameters
    ----------
    bulk_llm : BaseRagasBulkLLM
        Bulk LLM that runs the collected prompts.
    collect_window : float, default 2.0
        Seconds without a new prompt before the collected prompts are sent.
        Batches finish in minutes or hours, so a short wait costs nothing. A larger
        window only matters when callers do slow work between calls.
    max_collect_wait : float, default 30.0
        Longest time prompts are held back while others keep arriving.
    send_temperature : bool, default True
        Send the temperature of each call to the endpoint. Turn it off for
        models that reject it, such as reasoning models.
    run_config : RunConfig, optional
        Defaults to an unbounded ``max_workers``, since concurrency is what
        fills a batch. Timeouts and retries of this config are not used; the
        bulk LLM has its own.
    """

    def __init__(
        self,
        bulk_llm: BaseRagasBulkLLM,
        collect_window: float = 2.0,
        max_collect_wait: float = 30.0,
        send_temperature: bool = True,
        run_config: t.Optional[RunConfig] = None,
    ):
        if collect_window < 0:
            raise ValueError("collect_window must not be negative")
        if max_collect_wait <= 0:
            raise ValueError("max_collect_wait must be positive")
        super().__init__()
        self.bulk_llm = bulk_llm
        self.collect_window = collect_window
        self.max_collect_wait = max_collect_wait
        self.send_temperature = send_temperature
        self.set_run_config(run_config or RunConfig(max_workers=-1))
        # Each event loop gets its own queue, because futures belong to one loop.
        self._collectors: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, _Collector
        ] = weakref.WeakKeyDictionary()

    def _collector(self) -> _Collector:
        loop = asyncio.get_running_loop()
        collector = self._collectors.get(loop)
        if collector is None:
            collector = self._collectors[loop] = _Collector(self)
        return collector

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        raise NotImplementedError(
            "BulkLLMAdapter is async only: batches are collected from concurrent "
            "async calls, so use agenerate_text or generate."
        )

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        options: t.Dict[str, t.Any] = {}
        if temperature is not None and self.send_temperature:
            options["temperature"] = temperature
        if stop:
            options["stop"] = list(stop)
        texts = await self._collector().request(prompt.to_string(), options, n)
        return LLMResult(generations=[[Generation(text=text) for text in texts]])

    async def generate(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # No tenacity retry or timeout here: the bulk LLM retries on its own, and a
        # batch legitimately takes far longer than a request timeout.
        if temperature is None:
            temperature = self.get_temperature(n)
        return await self.agenerate_text(
            prompt=prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bulk_llm={self.bulk_llm.__class__.__name__})"
