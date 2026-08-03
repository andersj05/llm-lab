import pytest
from llm_lab.schemas import GenerationOptions
from pydantic import ValidationError


def test_generation_budget_supports_long_local_runs() -> None:
    assert GenerationOptions(max_new_tokens=32_768).max_new_tokens == 32_768

    with pytest.raises(ValidationError):
        GenerationOptions(max_new_tokens=32_769)
