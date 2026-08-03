# LLM Lab

LLM Lab is a local-first observability workbench for running open-weight language
models in Linux and inspecting what happens across tokens and layers. The first
supported runtime is the local Qwen3 0.6B model at
`/home/base/ai/qwen3-0.6b/model`.

The project deliberately separates four kinds of evidence:

- generated output;
- explicit scratchpad text, when a model is configured to emit it;
- internal activation and logit summaries;
- Jacobian-lens (J-space) concept readouts.

J-space readouts are approximate interpretability measurements, not a complete
or authoritative transcript of model reasoning.

## Repository layout

```text
apps/web/            Next.js observability dashboard
services/api/        FastAPI control plane and model worker
packages/contracts/  Shared event and API contracts
docs/                Architecture and operating notes
runs/                Local experiment artifacts (ignored by Git)
```

## Branching

- `main`: release-ready history
- `dev`: integration branch
- `feat/<name>`: focused work branched from `dev`

Changes land on a feature branch first, then merge into `dev`, and finally into
`main` when a release is ready.

## Local development

Prerequisites:

- Ubuntu 24.04 under WSL2
- Python 3.12 and `uv`
- Node.js 24 and npm 11
- NVIDIA CUDA support for GPU inference (CPU fallback is supported)

Copy the environment template and install both workspaces:

```bash
cp .env.example .env
make bootstrap
make fit-jlens
make dev
```

`make fit-jlens` is a one-time, GPU-intensive calibration step. It fits the
official Anthropic Jacobian-lens implementation to 100 local Qwen prompts and
writes `jacobian_lens.pt` beside the model directory, outside Git. The command
is resumable through `jacobian_lens.fit.pt`; rerunning it continues from the
last saved prompt. Its 64-token calibration window is sized for the local 8 GB
GPU. You can start the application before the fit finishes, but
J-space stays disabled until a compatible checkpoint is present.

Then open `http://localhost:3000` from Windows. The API listens on
`http://localhost:8000` by default.

Qwen3 0.6B exposes a 40,960-token context window. LLM Lab accepts generation
budgets up to 32,768 new tokens and clamps them to the space remaining after the
rendered prompt. Activation and attention capture remain bounded to the final
`LLM_LAB_TRACE_TOKEN_LIMIT` tokens so a long generation is not rejected just
because tracing is enabled.

Finished runs can be deleted individually or cleared from the History sidebar.
Deletion removes the SQLite record, its ordered events, and the corresponding
artifact directory.

The Layer activity view shows the size of each layer's residual-stream update
for each token. Color is normalized independently within a layer so a single
high-magnitude layer cannot flatten every other row. Switch to Residual
magnitude to inspect the total residual-stream norm instead.

J-space is a separate, model-specific readout. It uses the fitted average
Jacobian to ask which vocabulary concepts each intermediate residual state
would contribute at the final layer. It is an experimental interpretability
measurement, not hidden chain-of-thought or evidence that the model is
conscious. See [docs/jspace.md](docs/jspace.md) for the method and controls.

See [docs/architecture.md](docs/architecture.md) for the component model and
[docs/development.md](docs/development.md) for setup and branch conventions.
