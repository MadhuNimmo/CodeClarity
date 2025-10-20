import json
import torch
import os
import pandas as pd
import argparse
from models.side.side_loader import load_side_model
from models.comet_loader import load_comet_model
from evaluation.evaluator import evaluate_json_folder

parser = argparse.ArgumentParser(description="Evaluate JSON summaries and produce a single CSV")
parser.add_argument("--config", type=str, default="config/evaluation.json", help="Path to config JSON")
parser.add_argument("--output_csv", type=str, default=None, help="Path to save combined CSV (overrides config)")
args = parser.parse_args()

try:
    with open(args.config) as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in {args.config}: {e}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
side_model, side_tokenizer, use_fallback_side = load_side_model(config["SIDE_CHECKPOINT"], DEVICE)
comet_model = load_comet_model()

json_folder = config["JSON_FOLDER"]
if not os.path.exists(json_folder):
    raise FileNotFoundError(f"JSON folder not found: {json_folder}")

output_dir = os.path.join(json_folder, "csv_outputs")
os.makedirs(output_dir, exist_ok=True)

if args.output_csv:
    output_csv_path = args.output_csv
else:
    eval_csv_name = config.get("EVAL_CSV")
    output_csv_folder = config.get("OUTPUT_CSV_FOLDER")
    if not output_csv_folder:
        os.makedirs(output_csv_folder, exist_ok=True)
    output_csv_path = os.path.join(output_csv_folder, eval_csv_name)

FINAL_COLUMNS = [
    "sample_id", "model_name", "programming_language", "length_bucket", "prompt_used", "code", "docstring,"                                                             
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
