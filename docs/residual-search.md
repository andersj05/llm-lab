# Residual-guided nonlinear search

The iterative search API turns one open problem into a persisted graph of ordinary LLM Lab runs.
It is intended for experiments where several representations should be explored before committing
to one chain of reasoning.

## Loop

Each search performs these phases:

1. **Idea generation** asks the model for diverse strategy niches and an executor prompt for each.
2. **Branch execution** submits every idea together. The dependency graph is parallel even though
   the single 8 GB GPU worker executes requests serially to avoid duplicate model loads and OOMs.
3. **Residual evaluation** asks a separate run for a numeric residual vector. Default dimensions
   are correctness, completeness, constraints, and evidence.
4. **Diverse archive** retains the lowest-residual candidate in each named niche.
5. **Trust-region update** fits a derivative-free local linear model of scalar residual score over
   three explicit controls: breadth, rigor, and novelty. Actual versus predicted improvement grows
   or shrinks the next radius.
6. **Semantic realization** gives target coordinates, parent output, and evaluator feedback back to
   the model, which realizes the numerical step as a new textual idea.

This is DFO-LS-inspired rather than an implementation of the DFO-LS solver. Arbitrary text is not a
smooth vector space. Restricting the local surrogate to explicit bounded controls makes its slope
testable; global semantic exploration remains discrete.

## Start a search

With the API running on port 8000:

    curl -X POST http://localhost:8000/api/v1/searches \
      -H 'content-type: application/json' \
      -d '{
        "problem": "Your open problem",
        "goal": "A concrete definition of success",
        "evaluator_rubric": "Domain-specific checks and hard constraints",
        "options": {
          "iterations": 6,
          "branches": 4,
          "branch_tokens": 1536,
          "idea_tokens": 768,
          "evaluator_tokens": 512,
          "solved_threshold": 0.08,
          "seed": 42
        }
      }'

Inspect or cancel it with:

    curl http://localhost:8000/api/v1/searches/SEARCH_ID
    curl -X POST http://localhost:8000/api/v1/searches/SEARCH_ID/cancel

Search state is stored under runs/searches/SEARCH_ID/state.json. Every idea and evaluator node is
also an ordinary persisted run, so its output and diagnostics remain inspectable through the
existing run endpoints.

## Evaluator design

The built-in evaluator is an LLM judge, useful for prototyping but not an objective oracle. For
serious experiments, make the rubric measurable and add a domain evaluator behind the same
ResidualEvaluation contract. Good residuals are decomposed and falsifiable, such as:

- unit-test failures by category;
- distance from a numeric target plus constraint violations;
- theorem-checker errors;
- retrieval support and contradiction counts;
- simulation mismatch across held-out cases.

Do not treat repeated self-approval as convergence. A candidate should be considered solved only
when an external verifier agrees or the residual components are independently measurable.

## Local 4B model

The checkpoint is installed at:

    /home/base/ai/qwen3-4b-thinking-2507/model

Setting LLM_LAB_LOAD_IN_4BIT=true loads it in NF4 through bitsandbytes. This preserves Transformers
access to hidden states and logits while fitting the 4B model on the RTX 4070 Laptop GPU. Start with
four branches, 1,024-1,536 tokens per branch, and diagnostics disabled for search nodes. Long
contexts and raw attention capture can still exceed 8 GB even with quantized weights.

