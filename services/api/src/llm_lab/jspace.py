from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class JacobianLensAdapter:
    """Loads and applies Anthropic's model-specific Jacobian lens."""

    def __init__(self, checkpoint_path: Path | None, token_limit: int, layer_stride: int) -> None:
        self.checkpoint_path = checkpoint_path
        self.token_limit = max(1, token_limit)
        self.layer_stride = max(1, layer_stride)
        self._lens: Any | None = None
        self._lens_model: Any | None = None
        self._meaningful_mask: Tensor | None = None
        self._loaded_signature: tuple[int, int] | None = None
        self._load_lock = threading.Lock()

    def inspect(self, config: dict[str, Any], model_path: Path | None = None) -> dict[str, Any]:
        path = self.checkpoint_path
        if path is None:
            return {
                "available": False,
                "reason": "Fit a model-specific lens with `make fit-jlens`",
                "method": "jacobian_lens",
            }
        if not path.is_file():
            return {
                "available": False,
                "reason": f"Lens checkpoint not found: {path}",
                "method": "jacobian_lens",
            }
        try:
            import jlens  # noqa: F401
        except ImportError:
            return {
                "available": False,
                "reason": "Install the model dependencies to enable the Jacobian lens",
                "method": "jacobian_lens",
            }

        metadata = self._metadata()
        expected_width = config.get("hidden_size")
        fitted_width = metadata.get("d_model")
        if expected_width and fitted_width and expected_width != fitted_width:
            return {
                "available": False,
                "reason": (
                    f"Lens width {fitted_width} does not match model width {expected_width}"
                ),
                "method": "jacobian_lens",
            }
        expected_architecture = (config.get("architectures") or [None])[0]
        fitted_architecture = metadata.get("architecture")
        expected_vocabulary = config.get("vocab_size")
        fitted_vocabulary = metadata.get("vocab_size")
        fitted_fingerprint = metadata.get("model_fingerprint")
        if fitted_architecture and fitted_architecture != expected_architecture:
            return {
                "available": False,
                "reason": (
                    f"Lens architecture {fitted_architecture} does not match "
                    f"{expected_architecture}"
                ),
                "method": "jacobian_lens",
            }
        if fitted_vocabulary and fitted_vocabulary != expected_vocabulary:
            return {
                "available": False,
                "reason": (
                    f"Lens vocabulary {fitted_vocabulary} does not match {expected_vocabulary}"
                ),
                "method": "jacobian_lens",
            }
        if (
            fitted_fingerprint
            and model_path is not None
            and fitted_fingerprint != self._model_fingerprint(model_path)
        ):
            return {
                "available": False,
                "reason": "Lens fingerprint does not match the configured model files",
                "method": "jacobian_lens",
            }
        return {
            "available": True,
            "reason": None,
            "method": "jacobian_lens",
            "prompts": metadata.get("n_prompts"),
            "layers": metadata.get("n_layers"),
        }

    def capture(
        self,
        model: Any,
        tokenizer: Any,
        sequences: Any,
    ) -> dict[str, Any]:
        import torch
        from jlens.hooks import ActivationRecorder

        lens, lens_model = self._load(model, tokenizer)
        window = sequences[:, -self.token_limit :].to(lens_model.input_device)
        source_layers = lens.source_layers
        layers = source_layers[:: self.layer_stride]
        if source_layers[-1] not in layers:
            layers.append(source_layers[-1])
        output_layer = lens_model.n_layers - 1
        if output_layer not in layers:
            layers.append(output_layer)
        layers = sorted(set(layers))

        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
            lens_model.forward(window)
            activations = {layer: recorder.activations[layer].detach() for layer in layers}

        display_mask = self._display_mask(tokenizer, model.config.vocab_size, torch)
        top_n = 5
        layer_readouts: list[list[list[dict[str, int | float | str]]]] = []
        for layer in layers:
            residual = activations[layer][0].float()
            if layer in lens.jacobians:
                residual = lens.transport(residual, layer)
            logits = lens_model.unembed(residual).float()
            masked_logits = logits.masked_fill(~display_mask.to(logits.device), float("-inf"))
            _, top_ids = masked_logits.topk(top_n, dim=-1)
            selected_logits = logits.gather(1, top_ids)
            ranks = torch.empty_like(top_ids)
            for index in range(top_n):
                ranks[:, index] = (logits > selected_logits[:, index].unsqueeze(-1)).sum(dim=-1)

            position_readouts: list[list[dict[str, int | float | str]]] = []
            for position in range(top_ids.shape[0]):
                tokens: list[dict[str, int | float | str]] = []
                for index in range(top_n):
                    token_id = int(top_ids[position, index].item())
                    tokens.append(
                        {
                            "token_id": token_id,
                            "text": tokenizer.decode(
                                [token_id], clean_up_tokenization_spaces=False
                            ).strip(),
                            "rank": int(ranks[position, index].item()),
                            "logit": float(selected_logits[position, index].item()),
                        }
                    )
                position_readouts.append(tokens)
            layer_readouts.append(position_readouts)
            del logits, masked_logits, selected_logits, top_ids, ranks

        token_ids = window[0].tolist()
        return {
            "method": "jacobian_lens",
            "lens_prompts": lens.n_prompts,
            "lens_width": lens.d_model,
            "layers": layers,
            "output_layer": output_layer,
            "tokens": [
                tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                for token_id in token_ids
            ],
            "token_ids": token_ids,
            "token_offset": int(sequences.shape[-1] - window.shape[-1]),
            "truncated": sequences.shape[-1] > window.shape[-1],
            "top_tokens": layer_readouts,
        }

    def _load(self, model: Any, tokenizer: Any) -> tuple[Any, Any]:
        with self._load_lock:
            if self.checkpoint_path is None:
                raise RuntimeError("No Jacobian lens checkpoint is configured")
            checkpoint_stat = self.checkpoint_path.stat()
            signature = (checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size)
            if (
                self._lens is not None
                and self._lens_model is not None
                and self._loaded_signature == signature
            ):
                return self._lens, self._lens_model

            import jlens

            lens = jlens.JacobianLens.load(str(self.checkpoint_path))
            lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
            if lens.d_model != lens_model.d_model:
                raise RuntimeError(
                    f"Lens width {lens.d_model} does not match model width {lens_model.d_model}"
                )
            invalid_layers = [
                layer for layer in lens.source_layers if layer >= lens_model.n_layers - 1
            ]
            if invalid_layers:
                raise RuntimeError(f"Lens contains invalid source layers: {invalid_layers}")
            self._lens = lens
            self._lens_model = lens_model
            self._loaded_signature = signature
            return lens, lens_model

    def _display_mask(self, tokenizer: Any, vocab_size: int, torch: Any) -> Tensor:
        if self._meaningful_mask is not None:
            return self._meaningful_mask
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False).strip()
            except Exception:
                continue
            if not decoded or "<|" in decoded:
                continue
            if all(
                character.isalnum() or (0 < index < len(decoded) - 1 and character in "'-’")
                for index, character in enumerate(decoded)
            ):
                mask[token_id] = True
        self._meaningful_mask = mask
        return mask

    def _metadata(self) -> dict[str, Any]:
        if self.checkpoint_path is None:
            return {}
        metadata_path = self.checkpoint_path.with_suffix(".meta.json")
        if not metadata_path.is_file():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _model_fingerprint(model_path: Path) -> str:
        digest = hashlib.sha256()
        for name in ("config.json", "tokenizer_config.json"):
            path = model_path / name
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()
