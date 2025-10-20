#!/usr/bin/env python3
"""
Translate English summaries to multiple target languages using Cohere (chat API, aya-expanse-8b).

- Reads a JSON ARRAY (each item has "summary_english").
- Adds:
    summary_english_to_chinese
    summary_english_to_french
    summary_english_to_spanish
    summary_english_to_portuguese
    summary_english_to_arabic
    summary_english_to_hindi
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import cohere

MODEL_NAME_DEFAULT = "c4ai-aya-expanse-8b"
TEMPERATURE_DEFAULT = 0.0
MAX_RETRIES_DEFAULT = 3
RETRY_BASE_SLEEP = 1.0  # seconds

TARGET_LANGS = {
    "chinese": "Chinese (Simplified)",
    "french": "French",
    "spanish": "Spanish",
    "portuguese": "Portuguese",
    "arabic": "Arabic",
    "hindi": "Hindi",
}

PROMPT_TEMPLATE = """You are a precise technical translator for software engineering content.

Translate the English code summary below into {target_lang}.

Requirements:
- Preserve meaning exactly; do NOT add, omit, or infer details.
- Keep code identifiers and API/type names verbatim.

English summary:
---
{english_summary}
---
"""

@dataclass
class Args:
    input: str
    output: str
    model: str = MODEL_NAME_DEFAULT
    temperature: float = TEMPERATURE_DEFAULT
    max_retries: int = MAX_RETRIES_DEFAULT
    limit: int = 0  # 0 = no limit


def make_client() -> cohere.Client:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set COHERE_API_KEY in your environment (or .env).")
    return cohere.Client(api_key)


def _clean_translation(s: str) -> str:
    t = s.strip()
    if t.lower().startswith("translation:"):
        t = t.split(":", 1)[1].strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    strip_chars = '"\'“”‘’`'  # straight + curly quotes + backtick
    return t.strip(strip_chars)



def translate_once(client: cohere.Client, model: str, prompt: str, temperature: float) -> str:
    """
    Works with Cohere v1 chat API. Falls back to other fields if needed.
    """
    resp = client.chat(model=model, message=prompt, temperature=temperature)
    # Primary
    text: Optional[str] = getattr(resp, "text", None)

    # Newer SDKs sometimes return structured message content
    if not text:
        msg = getattr(resp, "message", None)
        parts = getattr(msg, "content", None)
        if parts:
            # parts can be list of objects with .text
            chunks = []
            for p in parts:
                pt = getattr(p, "text", None)
                if pt:
                    chunks.append(pt)
                elif isinstance(p, dict) and "text" in p:
                    chunks.append(p["text"])
            if chunks:
                text = " ".join(chunks)

    # Older generations API fallback (unlikely, but safe)
    if not text and hasattr(resp, "generations"):
        gens = getattr(resp, "generations", [])
        if gens:
            text = getattr(gens[0], "text", None)

    if not text:
        raise RuntimeError("Empty translation from API.")
    return _clean_translation(text)


def translate_with_retries(client: cohere.Client, model: str, prompt: str, temperature: float, max_retries: int) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return translate_once(client, model, prompt, temperature)
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
    raise RuntimeError("Exhausted retries unexpectedly.")


def process_file(args: Args) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    client = make_client()
    items = data if args.limit <= 0 else data[: args.limit]

    for item in tqdm(items, desc="Translating", unit="item"):
        eng = item.get("summary_english", "")
        if not eng or not eng.strip():
            continue

        for lang_key, lang_label in TARGET_LANGS.items():
            out_field = f"summary_english_to_{lang_key}"
            if out_field in item and str(item[out_field]).strip():
                continue

            prompt = PROMPT_TEMPLATE.format(target_lang=lang_label, english_summary=eng.strip())
            try:
                translated = translate_with_retries(
                        client=client,
                        model=args.model,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_retries=args.max_retries,  # e.g., 3
                )
                item[out_field] = translated
                #print(translated)
            except Exception as e:
                # Leave empty and log; continue to next language/sample
                item[out_field] = ""
                print(f"[WARN] translation failed for id={item.get('id')} lang={lang_key}: {e}", file=sys.stderr, flush=True)


    if args.limit > 0 and args.limit < len(data):
        data[: args.limit] = items
    else:
        data = items

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON array file.")
    parser.add_argument(
        "--output",
        help="Path to output JSON file. Defaults to <input_basename>_with_pivot_translations.json",
    )
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT, help=f"Cohere model (default: {MODEL_NAME_DEFAULT})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT, help="Sampling temperature.")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT, help="Max retries on API errors.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process first N items.")
    args_ns = parser.parse_args()

    base, ext = os.path.splitext(args_ns.input)
    out_path = args_ns.output or (base + "_with_pivot_translations.json")

    args = Args(
        input=args_ns.input,
        output=out_path,
        model=args_ns.model,
        temperature=args_ns.temperature,
        max_retries=args_ns.max_retries,
        limit=args_ns.limit,
    )
    process_file(args)
    print(f" Wrote: {args.output}")


if __name__ == "__main__":
    main()
