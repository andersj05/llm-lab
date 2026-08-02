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
make dev
```

Then open `http://localhost:3000` from Windows. The API listens on
`http://localhost:8000` by default.

See [docs/architecture.md](docs/architecture.md) for the component model and
[docs/development.md](docs/development.md) for setup and branch conventions.
testing.
