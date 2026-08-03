from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class ArtifactStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.data_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: str, name: str, value: Any) -> Path:
        path = self.run_dir(run_id) / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def write_tensors(self, run_id: str, tensors: dict[str, Tensor]) -> Path:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError(
                "Raw tensor persistence requires the 'model' dependency extra"
            ) from error

        path = self.run_dir(run_id) / "traces.safetensors"
        safe_tensors = {
            name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()
        }
        save_file(safe_tensors, path)
        return path

    def resolve_artifact(self, run_id: str, name: str) -> Path | None:
        if Path(name).name != name:
            return None
        path = self.data_dir / run_id / name
        if not path.is_file() or self.data_dir.resolve() not in path.resolve().parents:
            return None
        return path

    def delete_run(self, run_id: str) -> bool:
        root = self.data_dir.resolve()
        path = (self.data_dir / run_id).resolve()
        if path.parent != root:
            raise ValueError("Invalid run identifier")
        if not path.exists():
            return False
        if not path.is_dir():
            raise ValueError("Run artifact path is not a directory")
        shutil.rmtree(path)
        return True
