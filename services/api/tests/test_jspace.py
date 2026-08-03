import json
import sys
from types import ModuleType

from llm_lab.jspace import JacobianLensAdapter


def test_jspace_requires_a_model_specific_checkpoint() -> None:
    status = JacobianLensAdapter(None, token_limit=64, layer_stride=1).inspect({})

    assert status == {
        "available": False,
        "reason": "Fit a model-specific lens with `make fit-jlens`",
        "method": "jacobian_lens",
    }


def test_jspace_rejects_checkpoint_metadata_for_another_model(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "jacobian_lens.pt"
    checkpoint.write_bytes(b"not loaded during inspection")
    checkpoint.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "d_model": 2048,
                "architecture": "DifferentForCausalLM",
                "vocab_size": 100,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "jlens", ModuleType("jlens"))

    status = JacobianLensAdapter(checkpoint, token_limit=64, layer_stride=1).inspect(
        {
            "hidden_size": 1024,
            "architectures": ["Qwen3ForCausalLM"],
            "vocab_size": 151936,
        }
    )

    assert status["available"] is False
    assert status["reason"] == "Lens width 2048 does not match model width 1024"


def test_jspace_rejects_a_different_model_fingerprint(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
    checkpoint = tmp_path / "jacobian_lens.pt"
    checkpoint.write_bytes(b"not loaded during inspection")
    checkpoint.with_suffix(".meta.json").write_text(
        json.dumps({"d_model": 8, "model_fingerprint": "different"}),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "jlens", ModuleType("jlens"))

    status = JacobianLensAdapter(checkpoint, token_limit=64, layer_stride=1).inspect(
        {"hidden_size": 8}, model_path
    )

    assert status["available"] is False
    assert status["reason"] == "Lens fingerprint does not match the configured model files"
