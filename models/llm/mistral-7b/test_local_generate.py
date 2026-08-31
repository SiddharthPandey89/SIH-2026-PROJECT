"""
Minimal local smoke test for Mistral 7B.

Requires real local weights at MISTRAL_MODEL_PATH (or the default
models/llm/mistral-7b/weights directory). Does not mock generation.

Usage (from the repository root):

    python models/llm/mistral-7b/test_local_generate.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

COMPONENT_DIR = Path(__file__).resolve().parent
if str(COMPONENT_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENT_DIR))

from engine import get_engine, shutdown_all  # noqa: E402
from exceptions import MistralError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Load local Mistral 7B and generate a short reply.")
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: Mistral 7B is running locally.",
        help="User prompt to send to the local model.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model-path", default=None, help="Override MISTRAL_MODEL_PATH.")
    args = parser.parse_args()

    engine = get_engine(model_path=args.model_path)
    health = engine.health()
    print("health:", health)
    if not health.get("ok"):
        print("FAILED: local Mistral 7B weights are missing or invalid.", file=sys.stderr)
        print(health.get("error") or "", file=sys.stderr)
        return 1

    try:
        engine.initialize()
        print(
            f"loaded: backend={engine._backend.name} device={engine._backend.device} "  # noqa: SLF001
            f"path={engine.model_path}"
        )
        reply = engine.generate_text(
            args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print("--- response ---")
        print(reply)
        print("--- end ---")
        if not reply.strip():
            print("FAILED: model returned an empty string.", file=sys.stderr)
            return 1
        return 0
    except MistralError:
        traceback.print_exc()
        return 1
    finally:
        shutdown_all()


if __name__ == "__main__":
    raise SystemExit(main())
