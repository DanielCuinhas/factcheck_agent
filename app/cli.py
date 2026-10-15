"""Usage: python -m app.cli "https://example.com/article"   (or pass raw text)"""
import asyncio
import argparse
import sys

from .graph import graph


async def main(arg: str, provider: str | None = None, model: str | None = None):
    state = {"limitations": [], "model_provider": provider, "model": model}
    state["url" if arg.startswith(("http://", "https://")) else "raw_text"] = arg
    out = await graph.ainvoke(state)
    print(out["report"].model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fact-check a URL or plain text.")
    parser.add_argument("input", help="Article URL or text to check")
    parser.add_argument("--provider", choices=("anthropic", "openai", "huggingface"))
    parser.add_argument("--model", help="Model ID for the selected provider")
    args = parser.parse_args()
    asyncio.run(main(args.input, args.provider, args.model))
