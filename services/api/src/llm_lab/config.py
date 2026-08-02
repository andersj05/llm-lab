from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path
    data_dir: Path
    database_path: Path
    device: str
    dtype: str
    cors_origins: tuple[str, ...]
    jlens_path: Path | None
    trace_token_limit: int

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        data_dir = Path(os.getenv("LLM_LAB_DATA_DIR", "./runs")).expanduser()
        jlens_value = os.getenv("LLM_LAB_JLENS_PATH")
        return cls(
            model_path=Path(
                os.getenv("LLM_LAB_MODEL_PATH", "/home/base/ai/qwen3-0.6b/model")
            ).expanduser(),
            data_dir=data_dir,
            database_path=Path(
                os.getenv("LLM_LAB_DATABASE_PATH", str(data_dir / "llm-lab.sqlite3"))
            ).expanduser(),
            device=os.getenv("LLM_LAB_DEVICE", "auto"),
            dtype=os.getenv("LLM_LAB_DTYPE", "auto"),
            cors_origins=_csv(os.getenv("LLM_LAB_CORS_ORIGINS", "http://localhost:3000")),
            jlens_path=Path(jlens_value).expanduser() if jlens_value else None,
            trace_token_limit=int(os.getenv("LLM_LAB_TRACE_TOKEN_LIMIT", "256")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
