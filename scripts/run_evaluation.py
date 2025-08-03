import json
import torch
from models.side.side_loader import load_side_model
from models.comet_loader import load_comet_model
from evaluation.evaluator import evaluate_json_folder

with open("configs/paths_config.json") as f:
    config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
side_model, side_tokenizer, use_fallback_side = load_side_model(config["SIDE_CHECKPOINT"], DEVICE)
comet_model = load_comet_model()

if __name__ == "__main__":
    evaluate_json_folder(config["JSON_FOLDER"], config["OUTPUT_CSV"], side_model, side_tokenizer, comet_model, use_fallback_side)
