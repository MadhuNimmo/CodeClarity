import argparse
import json
from pathlib import Path
from generation.pipeline import run_pipeline

def load_generation_cfg(cfg_path: str):
    p = Path(cfg_path)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {
        "default_model": "gemma",
        "default_model_ids": {
            "codegemma": "google/codegemma-7b-it",
            "gemma": "google/gemma-2-9b-it",
            "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct"
        },
        "split": "test",
        "out_dir": "data/code_summaries/generated",
        "samples_per_bucket": 3,
        "target_natural_languages": ["English","Chinese","French","Spanish","Portuguese","Arabic","Hindi"]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/generation.json")
    ap.add_argument("--model", choices=["codegemma", "gemma", "qwen", "deepseek"])
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--split", default=None, help="codesearchnet split (train/valid/test)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--samples_per_bucket", type=int, default=None)
    ap.add_argument("--languages", nargs="*", default=None, help="Override natural languages (e.g., English Hindi)")
    ap.add_argument("--source", choices=["codesearchnet", "file"], default="codesearchnet",
                    help="Load from CodeSearchNet or a custom file")
    ap.add_argument("--input_file", default=None, help="Path to JSON/JSONL with code rows when --source file")
    args = ap.parse_args()

    cfg = load_generation_cfg(args.config)

    model_key = args.model or cfg.get("default_model", "gemma")
    defaults = cfg.get("default_model_ids", {})
    model_id = args.model_id or defaults.get(model_key)
    split = args.split or cfg.get("split", "test")
    out_dir = args.out_dir or cfg.get("out_dir", "data/code_summaries/generated")
    spb = args.samples_per_bucket or cfg.get("samples_per_bucket", 3)
    langs = args.languages or cfg.get("target_natural_languages")

    run_pipeline(
        model_key=model_key,
        model_id=model_id,
        split=split,
        out_dir=out_dir,
        samples_per_bucket=spb,
        target_natural_languages=langs,
        source=args.source,
        input_file=args.input_file
    )

if __name__ == "__main__":
    main()
