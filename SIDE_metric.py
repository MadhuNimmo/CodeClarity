import json
import csv
import os
from googletrans import Translator
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import torch.nn.functional as F
from tqdm import tqdm

# CONFIG
DATA_DIR = "data" 
OUTPUT_CSV = "side_summary_scores.csv"
CHECKPOINT = "Model/baseline/103080"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD SIDE MODEL
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModel.from_pretrained(CHECKPOINT).to(DEVICE)

# POOLING
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

# TRANSLATOR
translator = Translator()

# COSINE SIMILARITY
def compute_side_score(code, summary):
    encoded = tokenizer([code, summary], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**encoded)
    pooled = mean_pooling(output, encoded["attention_mask"])
    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return util.pytorch_cos_sim(normalized[0], normalized[1]).item()

# CSV
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["language", "code", "original_summary", "translated_summary", "SIDE_score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            language = filename.replace(".json", "")
            print(f"Evaluating language: {language}")

            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                entries = json.load(f)

            for item in tqdm(entries):
                code = item.get("code", "")
                summary = item.get("summary", "")

                try:
                    translated = translator.translate(summary, dest='en').text
                except:
                    translated = summary  # fallback

                score = compute_side_score(code, translated)

                writer.writerow({
                    "language": language,
                    "code": code,
                    "original_summary": summary,
                    "translated_summary": translated,
                    "SIDE_score": round(score, 4)
                })

