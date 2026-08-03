#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from llm_lab.config import Settings

JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def parse_args() -> argparse.Namespace:
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="Fit Anthropic's Jacobian lens for the configured local model."
    )
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--prompts", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    return parser.parse_args()


def collect_prompts(
    dataset_name: str,
    dataset_config: str,
    count: int,
    seed: int,
) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split="train")
    texts = [
        str(row["text"]).strip()
        for row in dataset
        if str(row.get("text", "")).strip() and not str(row["text"]).strip().startswith("=")
    ]
    random.Random(seed).shuffle(texts)

    prompts: list[str] = []
    buffer: list[str] = []
    word_count = 0
    for text in texts:
        buffer.append(text)
        word_count += len(text.split())
        if word_count < 180:
            continue
        prompts.append("\n\n".join(buffer))
        buffer = []
        word_count = 0
        if len(prompts) == count:
            break
    if len(prompts) < count:
        raise RuntimeError(f"Dataset produced only {len(prompts)} usable prompt groups")
    return prompts


def model_fingerprint(model_path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "tokenizer_config.json"):
        path = model_path / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.prompts < 1:
        raise ValueError("--prompts must be at least 1")

    import jlens
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = Settings.from_env()
    model_path = args.model_path.expanduser().resolve()
    prompts = collect_prompts(args.dataset, args.dataset_config, args.prompts, args.seed)
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto" if device == "cuda" else torch.float32,
        attn_implementation="eager",
    ).to(device)
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    source_layers = list(range(lens_model.n_layers - 1))

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_path.parent / "jacobian_lens.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".fit.pt")
    jlens.configure_logging()
    lens = jlens.fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.sequence_length,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    lens.save(str(output))

    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    metadata: dict[str, Any] = {
        "method": "jacobian_lens",
        "implementation": "anthropics/jacobian-lens",
        "implementation_revision": JLENS_REVISION,
        "model_path": str(model_path),
        "model_fingerprint": model_fingerprint(model_path),
        "d_model": lens.d_model,
        "architecture": (model_config.get("architectures") or [None])[0],
        "vocab_size": model_config.get("vocab_size"),
        "n_layers": len(lens.source_layers),
        "source_layers": lens.source_layers,
        "n_prompts": lens.n_prompts,
        "sequence_length": args.sequence_length,
        "skip_first": args.skip_first,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"lens": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
