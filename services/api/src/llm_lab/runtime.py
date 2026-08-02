from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from llm_lab.config import Settings
from llm_lab.schemas import CreateRunRequest, ModelInfo, TokenTrace

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(slots=True)
class GenerationResult:
    output_text: str
    tokens: list[TokenTrace]
    trace_summary: dict[str, Any] = field(default_factory=dict)
    raw_tensors: dict[str, Tensor] = field(default_factory=dict)


class Runtime(Protocol):
    def inspect(self) -> ModelInfo: ...

    def generate(
        self, request: CreateRunRequest, on_text: Callable[[str], None]
    ) -> GenerationResult: ...


class ModelRuntime:
    """Lazy, single-model Hugging Face runtime intended to be queue-owned."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()

    def inspect(self) -> ModelInfo:
        config_path = self.settings.model_path / "config.json"
        available = config_path.is_file()
        config: dict[str, Any] = {}
        if available:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        weight_bytes = sum(
            path.stat().st_size for path in self.settings.model_path.glob("*.safetensors")
        )
        jlens_available, jlens_reason = self._inspect_jlens()
        return ModelInfo(
            path=str(self.settings.model_path),
            available=available,
            loaded=self._model is not None,
            architecture=(config.get("architectures") or [None])[0],
            parameter_dtype=config.get("torch_dtype"),
            layers=config.get("num_hidden_layers"),
            hidden_size=config.get("hidden_size"),
            vocabulary_size=config.get("vocab_size"),
            weight_bytes=weight_bytes or None,
            device=self._device,
            jspace_available=jlens_available,
            jspace_reason=jlens_reason,
        )

    def _inspect_jlens(self) -> tuple[bool, str | None]:
        if self.settings.jlens_path is None:
            return False, "Set LLM_LAB_JLENS_PATH to a fitted compatible lens checkpoint"
        if not self.settings.jlens_path.exists():
            return False, f"Lens checkpoint not found: {self.settings.jlens_path}"
        try:
            import jlens  # noqa: F401
        except ImportError:
            return False, "Install the Anthropic jacobian-lens reference package"
        return (
            False,
            "The fitted lens was found, but the J-lens execution adapter is not enabled yet",
        )

    def _load(self) -> tuple[Any, Any, Any]:
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                import torch

                return torch, self._model, self._tokenizer

            if not (self.settings.model_path / "config.json").is_file():
                raise FileNotFoundError(f"Model not found at {self.settings.model_path}")

            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Model execution dependencies are missing; run `uv sync --extra model`"
                ) from error

            device = self.settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            dtype: Any = "auto"
            if self.settings.dtype != "auto":
                dtype = getattr(torch, self.settings.dtype)
            elif device == "cpu":
                dtype = torch.float32

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_path, local_files_only=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_path,
                local_files_only=True,
                dtype=dtype,
                attn_implementation="eager",
            ).to(device)
            self._model.eval()
            self._device = device
            return torch, self._model, self._tokenizer

    def generate(
        self, request: CreateRunRequest, on_text: Callable[[str], None]
    ) -> GenerationResult:
        torch, model, tokenizer = self._load()
        if request.traces.jspace and not self.inspect().jspace_available:
            raise RuntimeError(self.inspect().jspace_reason or "J-space is unavailable")

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=request.generation.enable_thinking,
            )
        except (AttributeError, TypeError, ValueError):
            rendered_prompt = "\n".join(message["content"] for message in messages)

        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {name: value.to(model.device) for name, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        if input_length + request.generation.max_new_tokens > self.settings.trace_token_limit:
            trace_requested = request.traces.hidden_states or request.traces.attentions
            if trace_requested:
                raise ValueError(
                    "Prompt plus generated token budget exceeds LLM_LAB_TRACE_TOKEN_LIMIT "
                    f"({self.settings.trace_token_limit})"
                )

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=60
        )
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": request.generation.max_new_tokens,
            "do_sample": request.generation.do_sample,
            "return_dict_in_generate": True,
            "output_scores": request.traces.logprobs,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if request.generation.do_sample:
            generation_kwargs.update(
                temperature=request.generation.temperature,
                top_p=request.generation.top_p,
            )

        torch.manual_seed(request.generation.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.generation.seed)

        result_box: list[Any] = []
        error_box: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    result_box.append(model.generate(**generation_kwargs))
            except BaseException as error:  # surfaced after the streamer exits
                error_box.append(error)

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()
        chunks: list[str] = []
        for chunk in streamer:
            chunks.append(chunk)
            on_text(chunk)
        generation_thread.join()
        if error_box:
            raise RuntimeError("Model generation failed") from error_box[0]
        if not result_box:
            raise RuntimeError("Model generation produced no result")

        generation_output = result_box[0]
        generated_ids = generation_output.sequences[0, input_length:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_traces = self._token_traces(
            torch, tokenizer, generated_ids, getattr(generation_output, "scores", None)
        )
        trace_summary, raw_tensors = self._capture_traces(
            torch,
            model,
            tokenizer,
            generation_output.sequences,
            request,
        )
        return GenerationResult(
            output_text=output_text,
            tokens=token_traces,
            trace_summary=trace_summary,
            raw_tensors=raw_tensors,
        )

    @staticmethod
    def _token_traces(
        torch: Any, tokenizer: Any, generated_ids: Any, scores: Any | None
    ) -> list[TokenTrace]:
        traces: list[TokenTrace] = []
        for index, token_tensor in enumerate(generated_ids):
            token_id = int(token_tensor.item())
            logprob: float | None = None
            alternatives: list[dict[str, int | float | str]] = []
            if scores is not None and index < len(scores):
                log_probs = torch.log_softmax(scores[index][0].float(), dim=-1)
                logprob = float(log_probs[token_id].item())
                values, ids = torch.topk(log_probs, k=5)
                alternatives = [
                    {
                        "token_id": int(candidate_id),
                        "text": tokenizer.decode([int(candidate_id)]),
                        "logprob": float(value),
                    }
                    for value, candidate_id in zip(values.tolist(), ids.tolist(), strict=True)
                ]
            traces.append(
                TokenTrace(
                    index=index,
                    token_id=token_id,
                    text=tokenizer.decode([token_id]),
                    logprob=logprob,
                    top_alternatives=alternatives,
                )
            )
        return traces

    @staticmethod
    def _capture_traces(
        torch: Any,
        model: Any,
        tokenizer: Any,
        sequences: Any,
        request: CreateRunRequest,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        if not request.traces.hidden_states and not request.traces.attentions:
            return {}, {}

        attention_mask = torch.ones_like(sequences, device=model.device)
        with torch.inference_mode():
            outputs = model(
                input_ids=sequences.to(model.device),
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=request.traces.hidden_states,
                output_attentions=request.traces.attentions,
                return_dict=True,
            )

        tokens = tokenizer.convert_ids_to_tokens(sequences[0].tolist())
        summary: dict[str, Any] = {"tokens": tokens}
        raw: dict[str, Tensor] = {}
        if request.traces.hidden_states and outputs.hidden_states:
            norms = torch.stack(
                [state[0].float().norm(dim=-1).cpu() for state in outputs.hidden_states]
            )
            summary["hidden_state_norms"] = norms.tolist()
            if request.traces.persist_raw:
                raw.update(
                    {
                        f"hidden.layer_{index:03d}": state[0].detach().cpu()
                        for index, state in enumerate(outputs.hidden_states)
                    }
                )
        if request.traces.attentions and outputs.attentions:
            mean_attention = torch.stack(
                [attention[0].float().mean(dim=0).cpu() for attention in outputs.attentions]
            )
            summary["mean_attentions"] = mean_attention.tolist()
            if request.traces.persist_raw:
                raw.update(
                    {
                        f"attention.layer_{index:03d}": attention[0].detach().cpu()
                        for index, attention in enumerate(outputs.attentions)
                    }
                )
        return summary, raw
