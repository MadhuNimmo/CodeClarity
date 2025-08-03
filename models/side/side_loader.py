import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def load_side_model(checkpoint_path, device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModel.from_pretrained(checkpoint_path).to(device).eval()
        print(f"✓ Loaded SIDE model from {checkpoint_path}")
        return model, tokenizer, False
    except Exception as e:
        print(f"Fallback to SentenceTransformer due to: {e}")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        return model, None, True
