# Development workflow

## Branches

Create work from the integration branch:

```bash
git switch dev
git pull --ff-only
git switch -c feat/<short-name>
```

Use `fix/`, `docs/`, and `chore/` for other focused change types. Merge feature
branches into `dev`; promote tested releases from `dev` into `main`. Do not
commit directly to `main`.

## Commit convention

Use short Conventional Commit subjects:

```text
feat(api): add queued model execution
feat(web): add layer-by-token trace matrix
test(api): cover run lifecycle transitions
ci: validate Python and web workspaces
```

## Local model

The default `.env.example` points to:

```text
/home/base/ai/qwen3-0.6b/model
```

Keep models outside this repository. Override `LLM_LAB_MODEL_PATH` in `.env`
when testing a different local model.

## Quality gates

Before committing:

```bash
make check
```

CI runs the same lint, type-check, test, and build checks without loading model
weights or requiring a GPU.

## CI and dependency updates

GitHub Actions runs independent API and web jobs on pushes to `main`, `dev`,
`feat/**`, and `fix/**`, and on pull requests into `main` or `dev`. The API job
installs only control-plane and test dependencies; it never downloads PyTorch or
model weights. The web job uses the committed npm lockfile and produces a full
Next.js production build.

Dependabot targets dependency updates at `dev` so automated updates follow the
same integration path as feature work.
