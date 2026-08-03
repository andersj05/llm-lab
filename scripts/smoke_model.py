from __future__ import annotations

import json

from llm_lab.config import Settings
from llm_lab.runtime import ModelRuntime
from llm_lab.schemas import CreateRunRequest, GenerationOptions, TraceOptions


def main() -> None:
    settings = Settings.from_env()
    runtime = ModelRuntime(settings)
    print(runtime.inspect().model_dump_json(indent=2))
    result = runtime.generate(
        CreateRunRequest(
            prompt="Reply with exactly the word ready.",
            generation=GenerationOptions(
                max_new_tokens=16,
                temperature=0,
                top_p=1,
                do_sample=False,
                enable_thinking=False,
            ),
            traces=TraceOptions(logprobs=False),
        ),
        lambda chunk: print(chunk, end="", flush=True),
    )
    print()
    print(json.dumps(result.generation, indent=2))


if __name__ == "__main__":
    main()

