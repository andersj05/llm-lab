# Architecture

LLM Lab is a Linux-first modular monolith. The browser is a visualization
client; model execution, instrumentation, persistence, and artifact generation
remain in the Python runtime where PyTorch can access the GPU directly.

## Runtime topology

```text
Windows browser
    |
    | HTTP + WebSocket over localhost
    v
Next.js web app (:3000)
    |
    | versioned JSON contracts
    v
FastAPI control plane (:8000)
    |
    +-- run coordinator (queue, cancellation, lifecycle events)
    +-- single model worker (lazy model loading, one GPU run at a time)
    +-- trace adapters (tokens, logits, activations, attention, J-lens)
    +-- artifact store (SQLite metadata + immutable files)
```

WSL2 forwards Linux ports to Windows localhost, so both development servers run
inside Ubuntu while the UI is viewed in an ordinary Windows browser.

## Boundaries

### Web application

`apps/web` owns presentation and interaction only. It may request runs, stream
events, query summaries, and download artifacts. It does not import model code,
load weights, or act as the durable source of experiment data.

### API and coordinator

`services/api` validates commands, persists lifecycle state, and serializes all
GPU work through a queue. Keeping a single process responsible for the model
prevents duplicate model loads and avoidable out-of-memory failures.

### Model runtime

The model is loaded lazily from `LLM_LAB_MODEL_PATH`. Model weights stay outside
the repository. CPU is a supported fallback; CUDA is selected automatically
when available.

### Trace adapters

Instrumentation is opt-in per run because raw hidden states and attention can
be much larger than generated text. Generation length is bounded by the model
context window, while trace capture uses a separate rolling tail window. This
keeps long runs available without allowing trace tensors to grow without bound.
Adapters return compact browser summaries and store full tensors as artifacts
when requested.

The Jacobian-lens adapter is optional. It requires both the reference `jlens`
package and a compatible fitted lens checkpoint. The system must report the
adapter as unavailable instead of substituting logit-lens output or fabricated
J-space values. The adapter verifies hidden width, vocabulary size, model
architecture, and fitted source layers before enabling the run control. During
a J-space run it records the configured token tail, transports each source
layer through its fitted average Jacobian, applies the model's own unembedding,
and returns ranked vocabulary concepts for the browser grid.

Layer activity and J-space answer different questions. Layer activity measures
the norm of a token's residual update (`||h[l+1] - h[l]||`) or total residual
magnitude (`||h[l]||`). J-space is a vocabulary readout of what an intermediate
state contributes after transport through the model-specific Jacobian lens.

## Persistence

SQLite stores searchable run metadata and event ordering. Large immutable
payloads live under `runs/<run-id>/`:

```text
manifest.json
events.jsonl
tokens.parquet
jspace.parquet
activations.safetensors
attentions.safetensors
```

The initial scaffold creates metadata and event storage. Tensor and Parquet
writers are introduced with the corresponding instrumentation adapters.

## Event contract

Every live event has a run ID, monotonically increasing sequence number,
timestamp, type, and typed payload. WebSockets carry lightweight events only;
large artifacts are downloaded over HTTP.

## Reproducibility

Every run records the model path and revision, tokenizer identity, random seed,
generation configuration, instrumentation flags, software versions, device,
dtype, and lens identity when present. A run is immutable after it reaches a
terminal state unless the user explicitly deletes it; deletion also removes its
events and artifact directory.

## Security

- Services bind to localhost by default.
- Model paths are server configuration, not arbitrary paths supplied by users.
- The API never serves arbitrary filesystem paths.
- Model weights, prompts, and run artifacts remain local unless explicitly
  exported.
- Git ignore rules prevent weights and artifacts from entering source control.
