import json
import os
import torch
import pandas as pd
from src.models.side.side_loader import load_side_model
from src.models.comet_loader import load_comet_model
from src.evaluation.evaluator import evaluate_json_folder
import argparse


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")


def setup_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_evaluation(config: dict, output_csv: str | None = None, save_dir: str | None = None):
    """Main evaluation routine."""
    device = setup_device()

    side_model, side_tokenizer, use_fallback_side = load_side_model(config["SIDE_CHECKPOINT"], device)
    comet_model = load_comet_model()
    json_folder = config["JSON_FOLDER"]
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"JSON folder not found: {json_folder}")

    # Determine output directory and CSV path
    output_dir = save_dir or config.get("OUTPUT_CSV_FOLDER") or os.path.join(json_folder, "csv_outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = output_csv or os.path.join(output_dir, config.get("EVAL_CSV", "evaluation_results.csv"))

    FINAL_COLUMNS = [
        "sample_id", "model_name", "programming_language", "length_bucket",
        "prompt_used", "code", "docstring",
        "bt_language", "reference_summary", "generated_summary", "backtranslated_summary",
        "bertscore_f1", "bleu", "chrf++", "rougeL", "meteor", "comet", "side",
    ]

    print(f"Processing all JSON files in {json_folder}...")

    df = evaluate_json_folder(json_folder, output_csv_path, side_model, side_tokenizer, comet_model, use_fallback_side)

    if df is not None and not df.empty:
        for col in FINAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[FINAL_COLUMNS]

        df.to_csv(output_csv_path, index=False)
        print(f"Combined CSV saved to: {output_csv_path}")
    else:
        print("No valid results found in the folder.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSON summaries and produce a single CSV")
    parser.add_argument("--config", type=str, default="config/evaluation.json", help="Path to config JSON file")
    parser.add_argument("--output_csv", type=str, default=None, help="Override output CSV path")
    parser.add_argument("--save_dir", type=str, default=None, help="Override directory to save CSVs")

    args = parser.parse_args()
    config = load_config(args.config)
    run_evaluation(config, output_csv=args.output_csv, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
