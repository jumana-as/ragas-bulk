"""
Bulk (asynchronous batch-endpoint) LLM support.

Requests are packed into as few batches as the provider's limits allow
(first-fit decreasing bin packing over request size, request count and
context tokens). Batches are submitted, polled until they finish, and every
request that failed is re-packed into new batches until it succeeds or runs
out of attempts. All state is appended to a JSONL checkpoint journal after every
change, so an interrupted run resumes without resubmitting finished or in-flight
work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import typing as t
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024
CHECKPOINT_VERSION = 2
DEFAULT_CHECKPOINT_PATH = Path(".ragas_bulk") / "checkpoint.json"
MAX_RETRY_DELAY = 60.0

_T = t.TypeVar("_T")


@dataclass(frozen=True)
class BulkLLMLimits:
    """
    Limits of one batch accepted by a bulk endpoint. ``None`` means unlimited.

    Parameters
    ----------
    max_requests : int, optional
        Maximum number of requests in one batch.
    max_request_mb : float, optional
        Maximum size of one batch in megabytes (1 MB = 1024 * 1024 bytes).
        The size is estimated from the JSON of each request.
    max_context_tokens : int, optional
        Maximum total prompt tokens in one batch.
    """

    max_requests: t.Optional[int] = None
    max_request_mb: t.Optional[float] = None
    max_context_tokens: t.Optional[int] = None

    def __post_init__(self):
        for name in ("max_requests", "max_request_mb", "max_context_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def max_request_bytes(self) -> t.Optional[int]:
        if self.max_request_mb is None:
            return None
        return int(self.max_request_mb * BYTES_PER_MB)


@dataclass
class BulkRequest:
    """One request that can be placed in a batch."""

    custom_id: str
    prompt: str
    body: t.Dict[str, t.Any]
    size_bytes: int
    tokens: int


@dataclass
class BulkBatchResult:
    """
    Result of polling a batch.

    ``status`` is ``"in_progress"`` while the batch is still running; any other
    value is terminal (for example ``"completed"``, ``"failed"``, ``"expired"``
    or ``"cancelled"``).
    """

    status: str
    responses: t.Dict[str, str] = field(default_factory=dict)
    errors: t.Dict[str, str] = field(default_factory=dict)
    error: t.Optional[str] = None
    input_tokens: t.Optional[int] = None
    output_tokens: t.Optional[int] = None
    cost: t.Optional[float] = None


@dataclass
class BulkRequestState:
    custom_id: str
    prompt: str
    status: str = "pending"  # pending | submitted | completed | failed
    attempts: int = 0
    batch_id: t.Optional[str] = None
    response: t.Optional[str] = None
    error: t.Optional[str] = None
    # Body fields for this request only, such as the temperature of one call.
    options: t.Dict[str, t.Any] = field(default_factory=dict)


@dataclass
class BulkBatchState:
    batch_id: str
    custom_ids: t.List[str]
    status: str = "submitted"  # submitted | completed | failed | expired | cancelled
    error: t.Optional[str] = None
    estimated_tokens: int = 0
    input_tokens: t.Optional[int] = None
    output_tokens: t.Optional[int] = None
    cost: t.Optional[float] = None
    submitted_at: t.Optional[float] = None


class BulkTransportError(RuntimeError):
    """
    Error raised by a transport method (``submit_batch`` or ``get_batch``).

    Set ``retryable=True`` only when retrying the same call is safe and may
    succeed, such as a rate limit or a server error on a read. Submit calls
    should only be marked retryable when the provider certainly created no
    batch, because a retry after an ambiguous failure can duplicate work.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: t.Optional[int] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class BulkGenerationError(RuntimeError):
    """Raised when some requests still have no response after all attempts."""

    def __init__(self, failed: t.Dict[str, str]):
        self.failed = failed
        details = ", ".join(f"{cid}: {err}" for cid, err in list(failed.items())[:5])
        super().__init__(
            f"{len(failed)} request(s) failed after all attempts ({details})"
        )


class BulkCheckpointMismatchError(ValueError):
    """Raised when a checkpoint was written for a different model or request settings."""


class BulkCheckpoint:
    """
    Append-only JSONL journal of requests and batches.

    The first line is a header with the format version and a fingerprint of the
    settings that shape the answers (model, endpoint, request body options). Each
    later line is one record, and a later record for an id replaces an earlier
    one. Changes are appended and flushed to disk, so a crash loses at most the
    line being written and the cost of a save does not grow with the run.

    One checkpoint file must be used by one LLM instance at a time.
    """

    #: Records beyond the live ones that are tolerated before the journal is rewritten.
    COMPACT_SLACK = 100

    def __init__(
        self, path: t.Optional[Path], fingerprint: t.Optional[str] = None
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.requests: t.Dict[str, BulkRequestState] = {}
        self.batches: t.Dict[str, BulkBatchState] = {}
        #: Request ids of submits found at load that never reported a batch id.
        self.interrupted_submits: t.List[t.List[str]] = []
        self._intents: t.Dict[str, t.List[str]] = {}
        self._queue: t.List[t.Tuple[str, str]] = []
        self._rewrite_pending = False

    @classmethod
    def load(
        cls,
        path: t.Optional[t.Union[str, os.PathLike]],
        fingerprint: t.Optional[str] = None,
    ) -> BulkCheckpoint:
        checkpoint = cls(Path(path) if path is not None else None, fingerprint)
        if checkpoint.path is None or not checkpoint.path.exists():
            return checkpoint

        lines = [
            line
            for line in checkpoint.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            checkpoint._rewrite_pending = True
            return checkpoint

        try:
            header = json.loads(lines[0])
        except ValueError:
            header = None
        if not isinstance(header, dict) or "version" not in header:
            raise ValueError(f"{checkpoint.path} is not a bulk LLM checkpoint")
        if header["version"] != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {header['version']} in {checkpoint.path}"
            )
        stored = header.get("fingerprint")
        if fingerprint is not None and stored is not None and stored != fingerprint:
            raise BulkCheckpointMismatchError(
                f"{checkpoint.path} was written for a different model or request "
                "settings, so its cached responses would not match this run. Use "
                "another checkpoint_path or delete the file."
            )
        if checkpoint.fingerprint is None:
            checkpoint.fingerprint = stored

        for number, line in enumerate(lines[1:], start=2):
            try:
                record = json.loads(line)
            except ValueError:
                if number == len(lines):
                    logger.warning(
                        "Ignoring a partial last line in %s (the previous run was "
                        "interrupted while writing)",
                        checkpoint.path,
                    )
                    checkpoint._rewrite_pending = True
                    break
                raise ValueError(
                    f"Corrupt checkpoint {checkpoint.path}: line {number} is not valid JSON"
                ) from None
            try:
                checkpoint._replay(record)
            except (KeyError, TypeError, AttributeError) as exc:
                raise ValueError(
                    f"Corrupt checkpoint {checkpoint.path}: bad record on line {number}"
                ) from exc

        if checkpoint._intents:
            checkpoint.interrupted_submits = list(checkpoint._intents.values())
            checkpoint._intents.clear()
            checkpoint._rewrite_pending = True
            logger.warning(
                "%d submit(s) in %s were interrupted before the provider returned a "
                "batch id (%d request(s)). The provider may hold a batch that this "
                "checkpoint does not track; those requests will be submitted again.",
                len(checkpoint.interrupted_submits),
                checkpoint.path,
                sum(len(ids) for ids in checkpoint.interrupted_submits),
            )
        live = len(checkpoint.requests) + len(checkpoint.batches)
        if len(lines) - 1 > 2 * live + cls.COMPACT_SLACK:
            checkpoint._rewrite_pending = True
        return checkpoint

    def _replay(self, record: t.Dict[str, t.Any]) -> None:
        kind = record["t"]
        if kind == "request":
            self.requests[record["d"]["custom_id"]] = BulkRequestState(**record["d"])
        elif kind == "batch":
            self.batches[record["d"]["batch_id"]] = BulkBatchState(**record["d"])
        elif kind == "drop":
            self.requests.pop(record["id"], None)
        elif kind == "intent":
            self._intents[record["token"]] = list(record["ids"])
        elif kind == "settled":
            self._intents.pop(record["token"], None)

    def touch_request(self, *custom_ids: str) -> None:
        """Queue the current state of requests for the next :meth:`save`."""
        self._queue.extend(("request", cid) for cid in custom_ids)

    def touch_batch(self, batch_id: str) -> None:
        """Queue the current state of a batch for the next :meth:`save`."""
        self._queue.append(("batch", batch_id))

    def drop_request(self, custom_id: str) -> None:
        if self.requests.pop(custom_id, None) is not None:
            self._queue.append(("drop", custom_id))

    def begin_submit(self, custom_ids: t.Sequence[str]) -> str:
        """Record that a submit is starting, before the provider can create a batch."""
        token = uuid.uuid4().hex
        self._intents[token] = list(custom_ids)
        self._queue.append(("intent", token))
        return token

    def end_submit(self, token: str) -> None:
        if self._intents.pop(token, None) is not None:
            self._queue.append(("settled", token))

    def _record(self, kind: str, key: str) -> t.Optional[t.Dict[str, t.Any]]:
        if kind == "request":
            request = self.requests.get(key)
            return None if request is None else {"t": kind, "d": asdict(request)}
        if kind == "batch":
            batch = self.batches.get(key)
            return None if batch is None else {"t": kind, "d": asdict(batch)}
        if kind == "drop":
            return {"t": kind, "id": key}
        if kind == "intent":
            ids = self._intents.get(key)
            return None if ids is None else {"t": kind, "token": key, "ids": ids}
        return {"t": kind, "token": key}

    def save(self) -> None:
        queue, self._queue = self._queue, []
        if self.path is None:
            self._rewrite_pending = False
            return
        if self._rewrite_pending or not self.path.exists():
            self._rewrite()
            return
        records = [self._record(kind, key) for kind, key in queue]
        lines = [json.dumps(r, ensure_ascii=False) for r in records if r is not None]
        if not lines:
            return
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _rewrite(self) -> None:
        """Write the whole state as a fresh journal, replacing the old file atomically."""
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records: t.List[t.Dict[str, t.Any]] = [
            {"t": "batch", "d": asdict(b)} for b in self.batches.values()
        ]
        records += [{"t": "request", "d": asdict(r)} for r in self.requests.values()]
        records += [
            {"t": "intent", "token": token, "ids": ids}
            for token, ids in self._intents.items()
        ]
        header = {"version": CHECKPOINT_VERSION, "fingerprint": self.fingerprint}
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in [header, *records])
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.path)
        self._rewrite_pending = False


@dataclass
class BulkUsage:
    """
    Totals across every batch and request recorded in a checkpoint.

    ``estimated_prompt_tokens`` comes from the local token count. ``input_tokens``
    and ``output_tokens`` are the counts reported by the provider and stay 0 when
    the provider does not report them.
    """

    batches_submitted: int = 0
    batches_completed: int = 0
    batches_failed: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    estimated_prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    @classmethod
    def from_checkpoint(cls, checkpoint: BulkCheckpoint) -> BulkUsage:
        usage = cls()
        for batch in checkpoint.batches.values():
            usage.batches_submitted += 1
            if batch.status == "completed":
                usage.batches_completed += 1
            elif batch.status != "submitted":
                usage.batches_failed += 1
            usage.estimated_prompt_tokens += batch.estimated_tokens
            usage.input_tokens += batch.input_tokens or 0
            usage.output_tokens += batch.output_tokens or 0
            usage.cost += batch.cost or 0.0
        for state in checkpoint.requests.values():
            if state.status == "completed":
                usage.requests_completed += 1
            elif state.status == "failed":
                usage.requests_failed += 1
        return usage


@dataclass
class _Bin:
    items: t.List[BulkRequest] = field(default_factory=list)
    size_bytes: int = 0
    tokens: int = 0

    def fits(self, request: BulkRequest, limits: BulkLLMLimits) -> bool:
        if (
            limits.max_requests is not None
            and len(self.items) + 1 > limits.max_requests
        ):
            return False
        max_bytes = limits.max_request_bytes
        if max_bytes is not None and self.size_bytes + request.size_bytes > max_bytes:
            return False
        if (
            limits.max_context_tokens is not None
            and self.tokens + request.tokens > limits.max_context_tokens
        ):
            return False
        return True

    def add(self, request: BulkRequest) -> None:
        self.items.append(request)
        self.size_bytes += request.size_bytes
        self.tokens += request.tokens


def pack_requests(
    requests: t.Sequence[BulkRequest], limits: BulkLLMLimits
) -> t.List[t.List[BulkRequest]]:
    """
    Pack requests into the fewest batches that respect ``limits``.

    Uses first-fit decreasing: requests are sorted by size, largest first, and
    each one goes into the first batch that still fits it on every dimension.
    A new batch is opened when none fits.

    Raises
    ------
    ValueError
        If a single request is larger than one batch can hold.
    """
    max_bytes = limits.max_request_bytes
    for request in requests:
        if max_bytes is not None and request.size_bytes > max_bytes:
            raise ValueError(
                f"Request {request.custom_id} is {request.size_bytes} bytes, "
                f"larger than the batch limit of {max_bytes} bytes"
            )
        if (
            limits.max_context_tokens is not None
            and request.tokens > limits.max_context_tokens
        ):
            raise ValueError(
                f"Request {request.custom_id} has {request.tokens} tokens, "
                f"more than the batch limit of {limits.max_context_tokens}"
            )

    bins: t.List[_Bin] = []
    ordered = sorted(requests, key=lambda r: (r.size_bytes, r.tokens), reverse=True)
    for request in ordered:
        for batch in bins:
            if batch.fits(request, limits):
                batch.add(request)
                break
        else:
            new_bin = _Bin()
            new_bin.add(request)
            bins.append(new_bin)
    return [batch.items for batch in bins]


def _in_flight_batches(checkpoint: BulkCheckpoint) -> t.List[BulkBatchState]:
    return [b for b in checkpoint.batches.values() if b.status == "submitted"]


class BaseRagasBulkLLM(ABC):
    """
    Base class for LLMs that run requests through an asynchronous batch endpoint.

    Subclasses implement the transport: :meth:`submit_batch` sends one batch and
    returns its provider id, and :meth:`get_batch` returns the status and
    per-request responses of a batch. Packing, polling, retries and
    checkpointing are handled here.

    Parameters
    ----------
    limits : BulkLLMLimits
        Per-batch limits of the bulk model.
    checkpoint_path : str or PathLike, optional
        JSONL journal used to persist requests and batches. Checkpointing is on
        by default at ``.ragas_bulk/checkpoint.json``. A rerun with the same
        prompts reuses finished responses, resumes in-flight batches and retries
        failed requests. The file holds the full prompts and responses, so keep
        it out of version control and delete it when the data is sensitive. It is
        tied to the model and request settings (see :meth:`fingerprint`); a file
        written for different settings is rejected with
        :class:`BulkCheckpointMismatchError`. Use one file per LLM instance. Pass
        None to keep state in memory only.
    poll_interval : float, default 30.0
        Seconds to wait between polls of unfinished batches.
    max_attempts : int, default 3
        Number of times one request is submitted before it is reported as failed.
    max_in_flight_batches : int, default 4
        Maximum number of batches that may be running at once across all
        concurrent :meth:`generate_bulk` calls on this instance. Submission waits
        until a running batch finishes, which keeps large runs under provider
        rate limits.
    max_batch_wait : float, optional
        Seconds a batch may stay unfinished before it is marked expired and its
        requests are retried in a new batch. None waits as long as the provider
        does. The provider keeps running an expired batch; no cancel is sent.
    max_transport_retries : int, default 5
        Retries of a retryable transport error (see :class:`BulkTransportError`),
        with exponential backoff, before the error is raised or counted as a
        failed poll.
    body_options : mapping, optional
        Extra fields merged into every request body, such as ``temperature``,
        ``max_tokens`` or ``response_format``. Set them to match the sync LLM, so
        bulk answers are produced with the same sampling settings.
    """

    #: Seconds before the first retry of a transport error; doubles on each retry.
    transport_backoff: float = 1.0
    #: Consecutive failed polls of one batch before the run gives up on it.
    max_poll_failures: int = 10

    def __init__(
        self,
        limits: BulkLLMLimits,
        checkpoint_path: t.Optional[
            t.Union[str, os.PathLike]
        ] = DEFAULT_CHECKPOINT_PATH,
        poll_interval: float = 30.0,
        max_attempts: int = 3,
        max_in_flight_batches: int = 4,
        max_batch_wait: t.Optional[float] = None,
        max_transport_retries: int = 5,
        body_options: t.Optional[t.Mapping[str, t.Any]] = None,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_in_flight_batches < 1:
            raise ValueError("max_in_flight_batches must be at least 1")
        if max_batch_wait is not None and max_batch_wait <= 0:
            raise ValueError("max_batch_wait must be positive")
        if max_transport_retries < 0:
            raise ValueError("max_transport_retries must not be negative")
        body_options = dict(body_options or {})
        reserved = sorted({"messages", "input"} & set(body_options))
        if reserved:
            raise ValueError(
                f"body_options must not set {reserved}; they hold the prompt"
            )
        self.limits = limits
        self.checkpoint_path = checkpoint_path
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self.max_in_flight_batches = max_in_flight_batches
        self.max_batch_wait = max_batch_wait
        self.max_transport_retries = max_transport_retries
        self.body_options = body_options
        self._checkpoint: t.Optional[BulkCheckpoint] = None
        self._poll_failures: t.Dict[str, int] = {}
        # Shared by every concurrent generate_bulk call on this instance. All
        # mutations run between awaits on one event loop, so no lock is needed;
        # these only coordinate work that spans an await.
        self._reserved = 0
        self._submitting: t.Set[str] = set()
        self._polling: t.Set[str] = set()

    def usage(self) -> BulkUsage:
        """Totals for every batch and request in this instance's checkpoint."""
        if self._checkpoint is None:
            return BulkUsage()
        return BulkUsage.from_checkpoint(self._checkpoint)

    def fingerprint(self) -> t.Dict[str, t.Any]:
        """
        Settings that shape the answers, stored with the checkpoint.

        A checkpoint is rejected when its fingerprint differs, so cached answers
        from another model or sampling setup are never reused. Subclasses add
        what identifies their model and endpoint. Do not include secrets: only a
        hash of this value is stored.
        """
        return {"body": self.build_body("\x00fingerprint\x00")}

    def _fingerprint_digest(self) -> str:
        text = json.dumps(self.fingerprint(), sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_checkpoint(self) -> BulkCheckpoint:
        path = Path(self.checkpoint_path) if self.checkpoint_path is not None else None
        if self._checkpoint is None or self._checkpoint.path != path:
            self._checkpoint = BulkCheckpoint.load(path, self._fingerprint_digest())
        return self._checkpoint

    def discard(self, custom_ids: t.Iterable[str]) -> None:
        """
        Forget finished requests so the next call asks the model again.

        Use it for answers that were received but turned out to be unusable;
        otherwise a rerun would replay the same cached answer.
        """
        checkpoint = self._checkpoint
        if checkpoint is None:
            return
        for cid in custom_ids:
            state = checkpoint.requests.get(cid)
            if state is not None and state.status in ("completed", "failed"):
                checkpoint.drop_request(cid)
        checkpoint.save()

    def count_tokens(self, text: str) -> int:
        """Estimate the tokens of a prompt. Override with a model tokenizer for exact counts."""
        return math.ceil(len(text) / 4)

    def build_body(self, prompt: str) -> t.Dict[str, t.Any]:
        """Build the request body sent to the endpoint for one prompt."""
        return {
            "messages": [{"role": "user", "content": prompt}],
            **self.body_options,
        }

    @abstractmethod
    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        """Submit one batch and return the provider's batch id."""
        ...

    @abstractmethod
    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        """Return the current status and per-request responses of a batch."""
        ...

    async def generate_bulk(
        self,
        prompts: t.Sequence[str],
        custom_ids: t.Optional[t.Sequence[str]] = None,
        raise_exceptions: bool = True,
        request_options: t.Optional[t.Sequence[t.Mapping[str, t.Any]]] = None,
    ) -> t.List[t.Optional[str]]:
        """
        Generate one response per prompt through packed batches.

        Parameters
        ----------
        prompts : sequence of str
            Prompts to run.
        custom_ids : sequence of str, optional
            Unique id per prompt, used to match results and to resume from the
            checkpoint. Defaults to the prompt index.
        raise_exceptions : bool, default True
            If True, raise :class:`BulkGenerationError` when any request has no
            response after all attempts. If False, return None for those.
        request_options : sequence of mapping, optional
            Body fields for each request alone, such as ``{"temperature": 1.0}``.
            They are merged over ``body_options``. A cached response is reused
            only when the prompt and the options are the same.

        Returns
        -------
        list of str or None
            Responses aligned with ``prompts``.
        """
        if custom_ids is None:
            custom_ids = [str(i) for i in range(len(prompts))]
        if len(custom_ids) != len(prompts):
            raise ValueError("custom_ids must have the same length as prompts")
        if len(set(custom_ids)) != len(custom_ids):
            raise ValueError("custom_ids must be unique")
        if request_options is not None and len(request_options) != len(prompts):
            raise ValueError("request_options must have the same length as prompts")
        options = [dict(o) for o in request_options or [{}] * len(prompts)]

        checkpoint = self._load_checkpoint()
        self._register_requests(checkpoint, custom_ids, prompts, options)
        checkpoint.save()

        while True:
            # Ids another concurrent call is submitting right now are left to it.
            pending_ids = [
                cid
                for cid in custom_ids
                if checkpoint.requests[cid].status == "pending"
                and cid not in self._submitting
            ]
            if pending_ids:
                pending = [
                    self._to_bulk_request(checkpoint.requests[cid])
                    for cid in pending_ids
                ]
                for group in pack_requests(pending, self.limits):
                    await self._submit_group(checkpoint, group)
            elif _in_flight_batches(checkpoint):
                await self._poll_in_flight(checkpoint)
            elif any(cid in self._submitting for cid in custom_ids):
                await asyncio.sleep(0)
            else:
                break

        responses: t.List[t.Optional[str]] = []
        failed: t.Dict[str, str] = {}
        for cid in custom_ids:
            state = checkpoint.requests[cid]
            if state.status == "completed":
                responses.append(state.response)
            else:
                responses.append(None)
                failed[cid] = state.error or "no response"

        self._log_usage(checkpoint)
        if failed:
            logger.warning(
                "Bulk generation finished with %d failed request(s)", len(failed)
            )
            if raise_exceptions:
                raise BulkGenerationError(failed)
        return responses

    def _log_usage(self, checkpoint: BulkCheckpoint) -> None:
        usage = BulkUsage.from_checkpoint(checkpoint)
        logger.info(
            "Bulk LLM usage: %d/%d batch(es) completed, %d failed; "
            "%d/%d request(s) completed, %d failed; "
            "~%d estimated prompt tokens, %d input and %d output tokens reported, "
            "cost %.6f",
            usage.batches_completed,
            usage.batches_submitted,
            usage.batches_failed,
            usage.requests_completed,
            usage.requests_completed + usage.requests_failed,
            usage.requests_failed,
            usage.estimated_prompt_tokens,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost,
        )

    def _register_requests(
        self,
        checkpoint: BulkCheckpoint,
        custom_ids: t.Sequence[str],
        prompts: t.Sequence[str],
        options: t.Sequence[t.Dict[str, t.Any]],
    ) -> None:
        for cid, prompt, request_options in zip(custom_ids, prompts, options):
            existing = checkpoint.requests.get(cid)
            if (
                existing is None
                or existing.prompt != prompt
                or existing.options != request_options
            ):
                checkpoint.requests[cid] = BulkRequestState(
                    custom_id=cid, prompt=prompt, options=request_options
                )
                checkpoint.touch_request(cid)
            elif existing.status == "failed":
                existing.status = "pending"
                existing.attempts = 0
                existing.error = None
                checkpoint.touch_request(cid)

    def request_error(self, custom_id: str) -> t.Optional[str]:
        """Why a request has no response, or None if it has one or is unknown."""
        if self._checkpoint is None:
            return None
        state = self._checkpoint.requests.get(custom_id)
        if state is None or state.status == "completed":
            return None
        return state.error or "no response"

    def _to_bulk_request(self, state: BulkRequestState) -> BulkRequest:
        body = {**self.build_body(state.prompt), **state.options}
        payload = {"custom_id": state.custom_id, "body": body}
        size_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return BulkRequest(
            custom_id=state.custom_id,
            prompt=state.prompt,
            body=body,
            size_bytes=size_bytes,
            tokens=self.count_tokens(state.prompt),
        )

    async def _submit_group(
        self, checkpoint: BulkCheckpoint, group: t.List[BulkRequest]
    ) -> None:
        # The cap counts batches of every concurrent call, plus submits in progress.
        while (
            len(_in_flight_batches(checkpoint)) + self._reserved
            >= self.max_in_flight_batches
        ):
            await self._poll_in_flight(checkpoint)
        # Waiting may have let another call take some of these requests.
        group = [
            r
            for r in group
            if checkpoint.requests[r.custom_id].status == "pending"
            and r.custom_id not in self._submitting
        ]
        if not group:
            return

        ids = [r.custom_id for r in group]
        self._reserved += 1
        self._submitting.update(ids)
        token = checkpoint.begin_submit(ids)
        checkpoint.save()
        try:
            batch_id = await self._call_with_retries(self.submit_batch, group)
        except BulkTransportError as exc:
            # A retryable error means the provider certainly created no batch.
            if exc.retryable:
                checkpoint.end_submit(token)
                checkpoint.save()
            raise
        finally:
            self._reserved -= 1
            self._submitting.difference_update(ids)

        checkpoint.end_submit(token)
        checkpoint.batches[batch_id] = BulkBatchState(
            batch_id=batch_id,
            custom_ids=ids,
            estimated_tokens=sum(r.tokens for r in group),
            submitted_at=time.time(),
        )
        checkpoint.touch_batch(batch_id)
        for request in group:
            state = checkpoint.requests[request.custom_id]
            state.status = "submitted"
            state.batch_id = batch_id
            state.attempts += 1
        checkpoint.touch_request(*ids)
        checkpoint.save()
        logger.info("Submitted batch %s with %d request(s)", batch_id, len(group))

    async def _poll_in_flight(self, checkpoint: BulkCheckpoint) -> None:
        """
        Check every running batch once, then wait before the next check.

        Batches another concurrent call is already checking are skipped, so a
        batch is never applied twice.
        """
        for batch in _in_flight_batches(checkpoint):
            if batch.batch_id in self._polling:
                continue
            self._polling.add(batch.batch_id)
            try:
                result = await self._check_batch(batch)
            finally:
                self._polling.discard(batch.batch_id)
            if result is None or batch.status != "submitted":
                continue
            self._apply_batch_result(checkpoint, batch, result)
            checkpoint.save()
        # Always yield, so concurrent calls that wait on this one can progress.
        await asyncio.sleep(self.poll_interval if _in_flight_batches(checkpoint) else 0)

    async def _check_batch(self, batch: BulkBatchState) -> t.Optional[BulkBatchResult]:
        """Return the batch result once it is final, otherwise None."""
        try:
            result = await self._call_with_retries(self.get_batch, batch.batch_id)
        except BulkTransportError as exc:
            if not exc.retryable:
                raise
            count = self._poll_failures.get(batch.batch_id, 0) + 1
            self._poll_failures[batch.batch_id] = count
            if count >= self.max_poll_failures:
                raise
            logger.warning(
                "Polling batch %s failed (%d/%d): %s",
                batch.batch_id,
                count,
                self.max_poll_failures,
                exc,
            )
            return None
        self._poll_failures.pop(batch.batch_id, None)
        if result.status == "in_progress":
            if self._timed_out(batch):
                logger.warning(
                    "Batch %s exceeded max_batch_wait; its requests will be retried "
                    "in a new batch. The provider keeps running it.",
                    batch.batch_id,
                )
                return BulkBatchResult(
                    status="expired",
                    error=f"batch exceeded max_batch_wait of {self.max_batch_wait}s",
                )
            return None
        return result

    def _timed_out(self, batch: BulkBatchState) -> bool:
        if self.max_batch_wait is None or batch.submitted_at is None:
            return False
        return time.time() - batch.submitted_at > self.max_batch_wait

    async def _call_with_retries(
        self, fn: t.Callable[..., t.Awaitable[_T]], *args: t.Any
    ) -> _T:
        """Call a transport method, retrying errors marked as retryable."""
        attempt = 0
        while True:
            try:
                return await fn(*args)
            except BulkTransportError as exc:
                if not exc.retryable or attempt >= self.max_transport_retries:
                    raise
                delay = min(self.transport_backoff * (2**attempt), MAX_RETRY_DELAY)
                attempt += 1
                logger.warning(
                    "Transient error from %s (retry %d/%d in %.1fs): %s",
                    getattr(fn, "__name__", "transport"),
                    attempt,
                    self.max_transport_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

    def _apply_batch_result(
        self,
        checkpoint: BulkCheckpoint,
        batch: BulkBatchState,
        result: BulkBatchResult,
    ) -> None:
        batch.status = result.status
        batch.error = result.error
        batch.input_tokens = result.input_tokens
        batch.output_tokens = result.output_tokens
        batch.cost = result.cost
        logger.info(
            "Batch %s finished with status %s (%d/%d responses)",
            batch.batch_id,
            result.status,
            len(result.responses),
            len(batch.custom_ids),
        )
        for cid in batch.custom_ids:
            state = checkpoint.requests.get(cid)
            if state is None or state.batch_id != batch.batch_id:
                continue
            checkpoint.touch_request(cid)
            if cid in result.responses:
                state.status = "completed"
                state.response = result.responses[cid]
                state.error = None
                continue
            state.error = (
                result.errors.get(cid)
                or result.error
                or (f"batch {result.status} without a response")
            )
            state.status = "pending" if state.attempts < self.max_attempts else "failed"
        checkpoint.touch_batch(batch.batch_id)
