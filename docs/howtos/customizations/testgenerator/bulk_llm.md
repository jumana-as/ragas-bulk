# Running Testset Generation Through a Batch Endpoint

Testset generation makes many LLM calls: summaries, themes and entities for every chunk, then personas, scenarios and one sample per row. Many providers offer a **batch endpoint** that runs a large set of requests asynchronously at a lower price. A bulk LLM sends your requests in packed batches instead of one call at a time.

Passing `bulk_llm` runs **every** LLM call of the generation this way.

## Example

```python
import os
from ragas.llms import BulkLLMLimits, InlineBatchBulkLLM
from ragas.testset.synthesizers.generate import TestsetGenerator

bulk_llm = InlineBatchBulkLLM(
    BulkLLMLimits(max_requests=5000, max_request_mb=100),
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    # Match the sampling settings of your regular LLM.
    body_options={"max_tokens": 4096},
    max_in_flight_batches=2,
)

generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
testset = generator.generate_with_chunks(
    chunks,
    testset_size=50,
    bulk_llm=bulk_llm,
)
```

`InlineBatchBulkLLM` works with any provider whose batch API takes the requests inline as JSON (`POST {base_url}/batches`, `GET {base_url}/batches/{id}`). For other protocols, subclass `BaseRagasBulkLLM` and implement `submit_batch` and `get_batch`.

`bulk_llm` works with `generate_with_chunks`, `generate_with_langchain_docs`, `generate_with_llamaindex_docs` and `generate`. The `llm` you gave the generator is not used for the calls that go through the bulk LLM.

## How the stages run

Each stage depends on the one before it, and a batch can take minutes or hours, so a run is a series of batch round trips:

| Stage | How it uses the bulk LLM |
| --- | --- |
| Summaries, themes, entities | All prompts are known up front and go out together. |
| Personas | One batch. |
| Scenarios | One or two batches per wave of nodes or clusters. |
| Samples | One batch for all rows. |

Calls that are made at about the same time share a batch. A `BulkLLMAdapter` collects them, and prompts, output parsing and the fix-output-format retries work as on the sync path. A fix retry is another round trip.

To tune the collection, pass an adapter instead of the bulk LLM:

```python
from ragas.llms import BulkLLMAdapter

bulk_llm = BulkLLMAdapter(
    bulk_llm,
    collect_window=2.0,      # seconds without a new prompt before a batch is sent
    send_temperature=False,  # for models that reject a temperature
)
```

The temperature of each call (for example `1.0` for personas) is sent with its request, and `body_options` carries the settings that are the same for every request.

Scenario generation used to await one LLM call after another. It now asks for the prompts of a wave together, only as many as are needed to reach `testset_size`, so the number of calls and the scenarios are the same as before. On the sync path the wave runs at most `llm.run_config.max_workers` calls at a time.

## What the bulk LLM does for you

- **Packing.** Requests are packed into as few batches as `BulkLLMLimits` allow (request count, size in MB, context tokens).
- **Rate limits.** `max_in_flight_batches` (default 4) caps the batches running at once across all stages.
- **Retries.** Transient transport errors are retried with exponential backoff, and polling tolerates repeated failures. `max_batch_wait` expires a stuck batch and resubmits its requests.
- **No timeouts.** The `run_config` timeout and retry settings do not apply to batch calls, and the number of concurrent workers is not capped, because concurrency is what fills a batch. Leave `batch_size` unset in `generate`.
- **Usage.** Batches, requests, tokens and provider-reported cost are logged after each round trip and available from `bulk_llm.usage()`.

## Checkpointing

State is appended to `.ragas_bulk/checkpoint.json` after every change. If a run is interrupted, running the same code again reuses finished responses and resumes in-flight batches instead of paying for them twice.

- Scenario selection is random, so a rerun would ask different prompts. With `bulk_llm`, `generate` seeds Python's and NumPy's generators from `run_config.seed` (default 42), so a rerun with the same chunks asks the same prompts and is answered from the checkpoint. This also means a second run with the same seed returns the same testset for free; change the seed, or delete the checkpoint, to draw a new one.
- The file holds the **full prompts and responses**, which include your chunk text. Keep it out of version control and delete it when the data is sensitive. Use `checkpoint_path=None` to keep state in memory only.
- A checkpoint is tied to the model, endpoint, batch options and `body_options` it was written with. Using it with other settings raises `BulkCheckpointMismatchError`; pass another `checkpoint_path` or delete the file.
- Use one checkpoint file per bulk LLM instance.
- If the process dies while a batch is being submitted, the provider may hold a batch that the checkpoint does not know about. The next run logs a warning with the number of affected requests and submits them again.
- A sample that failed to parse is cached like any other answer. Rerunning replays it; delete the checkpoint (or call `bulk_llm.discard(ids)`) to ask again.

## Failure behavior

As on the sync path, one extraction input that stays unusable after all retries fails the extraction. To keep the rest of the run instead, build your own transforms and set `bulk_raise_exceptions=False` on the extractors; the affected nodes then get an empty value (`None` for summaries, `[]` for themes and entities). During scenario and sample generation, a request that still has no response fails its own task, and `raise_exceptions` of `generate` decides whether that stops the run.

```python
from ragas.testset.transforms import SummaryExtractor

extractor = SummaryExtractor(llm=llm, bulk_llm=bulk_llm, bulk_raise_exceptions=False)
```

`transforms_bulk_llm` still sets a bulk LLM for the default extractors alone, and cannot be combined with `transforms`. `bulk_llm` cannot be combined with `transforms_llm` or `return_executor`.

## Security notes

- `base_url` must use `https`. Plain `http` is only accepted for `localhost`.
- Redirects are not followed, so the API key is never sent to another host.
- The API key is sent only in the `Authorization` header and is never stored in the checkpoint.
- Results stay on the provider for its retention period (30 days for OpenRouter). Delete them there if needed.
