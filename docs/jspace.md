# J-space in LLM Lab

LLM Lab implements the Jacobian-lens method from Anthropic's Global Workspace
work. For each source layer, the calibration job averages the Jacobian of the
final residual stream with respect to that layer's residual stream over a
prompt corpus. At run time, the UI applies that fitted linear transport to each
captured token state and decodes the result with Qwen's own output head.

This differs from a logit lens. A logit lens decodes an intermediate state
directly; the Jacobian lens first estimates how the remaining layers transform
that state. The resulting tokens are model-specific concept readouts, not a
literal transcript of private reasoning.

## Fit the local Qwen lens

From Ubuntu, with the Qwen3 0.6B model available at the configured model path:

```bash
make fit-jlens
```

Defaults:

- 100 WikiText prompts, each 128 tokens long;
- all source layers through the layer before the final output layer;
- calibration batches of 16 hidden dimensions;
- a checkpoint saved every five prompts;
- final lens at `/home/base/ai/qwen3-0.6b/jacobian_lens.pt`;
- resumable accumulator at
  `/home/base/ai/qwen3-0.6b/jacobian_lens.fit.pt`.

The model path and output can be overridden explicitly:

```bash
uv run --extra model --extra lens-fit python scripts/fit_jlens.py \
  --model-path /path/to/model \
  --output /path/to/jacobian_lens.pt
```

The `.meta.json` sidecar records the model fingerprint, architecture, hidden
width, vocabulary size, corpus, prompt count, and fitting settings. The API
uses it to reject a lens fitted for a different model.

## Read the viewer

Columns are source tokens and rows are model layers. Each cell shows the
highest-ranked word-like vocabulary concept decoded from that layer and token.
Color represents its rank in the full vocabulary: stronger cells rank closer
to first. Select a cell to inspect the top five concepts and exact ranks.

The final `Output` row is the ordinary model output-head readout. It is included
as a reference point and is not transformed by a fitted Jacobian.

By default J-space captures the final 64 tokens and every fitted layer. Control
the amount of work with:

```dotenv
LLM_LAB_JSPACE_TOKEN_LIMIT=64
LLM_LAB_JSPACE_LAYER_STRIDE=1
```

J-space adds an additional instrumented forward pass after generation and can
be substantially slower than a normal run. Long prompts or generations are
shown as a labeled tail window rather than silently exceeding the configured
capture bound.

## Interpretation limits

The fitted Jacobian is a corpus average and a first-order approximation. Its
readouts can be useful for comparing where concepts become available across
layers, but they can also be unstable, tokenization-dependent, or misleading.
Do not present them as hidden chain-of-thought, a causal proof of why the model
answered, or evidence of consciousness.

Primary references:

- Anthropic, “A global workspace for large language models”:
  <https://www.anthropic.com/research/global-workspace>
- Method and experiments:
  <https://transformer-circuits.pub/2026/workspace/index.html>
- Reference implementation:
  <https://github.com/anthropics/jacobian-lens>
