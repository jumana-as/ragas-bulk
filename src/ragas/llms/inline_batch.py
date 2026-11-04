"""
Bulk LLM for providers whose batch API takes the requests inline as JSON.

The provider receives one POST with every request in the body, returns a batch
id, and later returns the results inline on a GET of that batch. This is not the
OpenAI file-upload flow, so the OpenAI client cannot be used for it.
"""

from __future__ import annotations

import asyncio
import json
import typing as t
import urllib.error
import urllib.parse
import urllib.request

from ragas.llms.bulk import (
    BaseRagasBulkLLM,
    BulkBatchResult,
    BulkLLMLimits,
    BulkRequest,
    BulkTransportError,
)

SUPPORTED_ENDPOINTS = ("/v1/chat/completions", "/v1/responses")

_NOT_FOUND_STATUSES = (404, 410)
_TERMINAL_STATUSES = ("failed", "expired", "cancelled")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects, so the API key is never forwarded to another host."""

    def redirect_request(self, *args: t.Any, **kwargs: t.Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _open(request: urllib.request.Request, timeout: float) -> t.Any:
    return _OPENER.open(request, timeout=timeout)


class InlineBatchBulkLLM(BaseRagasBulkLLM):
    """
    Bulk LLM that submits batches to an inline-JSON batch API over HTTP.

    Batches are created with ``POST {base_url}/batches`` and read with
    ``GET {base_url}/batches/{id}``. Requests are sent with the standard library
    in a worker thread, so no extra dependency is needed.

    Parameters
    ----------
    limits : BulkLLMLimits
        Per-batch limits of the provider.
    model : str
        Model name sent with every batch, for example ``"openai/gpt-4o-mini"``.
    base_url : str
        API root that the batch paths are appended to, for example
        ``"https://openrouter.ai/api/v1"``. Must be https; plain http is only
        accepted for localhost. Redirects are not followed.
    api_key : str
        Bearer token. It is sent only in the Authorization header.
    endpoint : str, default "/v1/chat/completions"
        Endpoint each request is sent to. One of ``"/v1/chat/completions"`` or
        ``"/v1/responses"``.
    batch_options : mapping, optional
        Extra top-level fields for the batch creation request, such as
        ``{"provider": {...}}`` or ``{"callback_url": "..."}``.
    headers : mapping, optional
        Extra HTTP headers sent with every call.
    request_timeout : float, default 120.0
        Socket timeout in seconds for one HTTP call.
    **kwargs
        Passed to :class:`BaseRagasBulkLLM` (``checkpoint_path``, ``poll_interval``,
        ``max_in_flight_batches``, ``max_batch_wait``, ``body_options``, and so on).
    """

    def __init__(
        self,
        limits: BulkLLMLimits,
        *,
        model: str,
        base_url: str,
        api_key: str,
        endpoint: str = "/v1/chat/completions",
        batch_options: t.Optional[t.Mapping[str, t.Any]] = None,
        headers: t.Optional[t.Mapping[str, str]] = None,
        request_timeout: float = 120.0,
        **kwargs: t.Any,
    ):
        if endpoint not in SUPPORTED_ENDPOINTS:
            raise ValueError(
                f"endpoint must be one of {SUPPORTED_ENDPOINTS}, got {endpoint!r}"
            )
        batch_options = dict(batch_options or {})
        if "requests" in batch_options:
            raise ValueError("batch_options must not contain 'requests'")
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in _LOCAL_HOSTS
        ):
            raise ValueError(
                "base_url must use https (http is only allowed for localhost), "
                f"got {base_url!r}"
            )
        super().__init__(limits, **kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.endpoint = endpoint
        self.batch_options = batch_options
        self.headers = dict(headers or {})
        self.request_timeout = request_timeout

    def build_body(self, prompt: str) -> t.Dict[str, t.Any]:
        if self.endpoint == "/v1/responses":
            return {"input": prompt, **self.body_options}
        return super().build_body(prompt)

    def fingerprint(self) -> t.Dict[str, t.Any]:
        return {
            **super().fingerprint(),
            "model": self.model,
            "endpoint": self.endpoint,
            "batch_options": self.batch_options,
        }

    async def submit_batch(self, requests: t.List[BulkRequest]) -> str:
        # The requests list goes last: the provider reads the body as a stream and
        # rejects the submit if the other fields come after it.
        payload: t.Dict[str, t.Any] = {
            "endpoint": self.endpoint,
            "model": self.model,
            "completion_window": "24h",
            **self.batch_options,
            "requests": [{"custom_id": r.custom_id, "body": r.body} for r in requests],
        }
        # A POST that fails with a server error or a dropped connection may still
        # have created the batch, so only rate limits are retried here.
        data = await self._send("POST", "/batches", payload, read=False)
        batch_id = data.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise BulkTransportError("batch submit response has no id")
        return batch_id

    async def get_batch(self, batch_id: str) -> BulkBatchResult:
        try:
            data = await self._send("GET", f"/batches/{batch_id}", None, read=True)
        except BulkTransportError as exc:
            if exc.status_code in _NOT_FOUND_STATUSES:
                return BulkBatchResult(
                    status="failed",
                    error=f"batch {batch_id} is no longer available (HTTP {exc.status_code})",
                )
            raise
        return self._parse_batch(data)

    async def _send(
        self,
        method: str,
        path: str,
        payload: t.Optional[t.Dict[str, t.Any]],
        *,
        read: bool,
    ) -> t.Dict[str, t.Any]:
        return await asyncio.to_thread(self._send_sync, method, path, payload, read)

    def _send_sync(
        self,
        method: str,
        path: str,
        payload: t.Optional[t.Dict[str, t.Any]],
        read: bool,
    ) -> t.Dict[str, t.Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            **self.headers,
        }
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method, headers=headers
        )
        label = f"{method} {path}"
        try:
            with _open(request, self.request_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if read:
                retryable = exc.code == 429 or exc.code >= 500
            else:
                retryable = exc.code == 429
            raise BulkTransportError(
                f"{label} failed with HTTP {exc.code}: {detail}",
                retryable=retryable,
                status_code=exc.code,
            ) from exc
        except OSError as exc:
            raise BulkTransportError(f"{label} failed: {exc}", retryable=read) from exc

        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError as exc:
            raise BulkTransportError(
                f"{label} returned invalid JSON", retryable=read
            ) from exc
        if not isinstance(body, dict):
            raise BulkTransportError(f"{label} returned a non-object JSON body")
        return body

    def _parse_batch(self, data: t.Dict[str, t.Any]) -> BulkBatchResult:
        status = str(data.get("status", ""))
        usage = data.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        cost = usage.get("cost")
        results = data.get("results")

        if status == "completed":
            # Results can be null for a short time after the status changes.
            if results is None:
                return BulkBatchResult(status="in_progress")
            out_status = "completed"
        elif status in _TERMINAL_STATUSES:
            out_status = status
        else:
            return BulkBatchResult(status="in_progress")

        responses: t.Dict[str, str] = {}
        errors: t.Dict[str, str] = {}
        for entry in results or []:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("custom_id"), str
            ):
                continue
            cid = entry["custom_id"]
            entry_error = _error_message(entry.get("error"))
            response = entry.get("response")
            if entry_error or not isinstance(response, dict):
                errors[cid] = entry_error or "no response"
                continue
            body = response.get("body") or {}
            status_code = response.get("status_code")
            if isinstance(status_code, int) and status_code >= 400:
                errors[cid] = (
                    f"HTTP {status_code}: {_error_message(body.get('error')) or body}"
                )
                continue
            if body.get("status") not in (None, "completed"):
                errors[cid] = f"response status {body.get('status')}"
                continue
            text = _response_text(body, self.endpoint)
            if text is None:
                errors[cid] = "response has no text"
            else:
                responses[cid] = text

        return BulkBatchResult(
            status=out_status,
            responses=responses,
            errors=errors,
            error=_error_message(data.get("error")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )


def _error_message(error: t.Any) -> t.Optional[str]:
    if not error:
        return None
    if isinstance(error, dict):
        message = error.get("message")
        return str(message) if message else json.dumps(error)
    return str(error)


def _response_text(body: t.Dict[str, t.Any], endpoint: str) -> t.Optional[str]:
    if endpoint == "/v1/responses":
        if isinstance(body.get("output_text"), str):
            return body["output_text"]
        parts: t.List[str] = []
        for item in body.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    parts.append(part["text"])
        return "".join(parts) if parts else None

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None
