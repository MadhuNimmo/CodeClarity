import torch
from sentence_transformers import util

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return summed / counts

def compute_side_score_batch(codes, hyps, side_model, side_tokenizer, device, use_fallback):
    sims = []
    if not use_fallback:
        with torch.no_grad():
            code_enc = side_tokenizer(codes, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            hyp_enc = side_tokenizer(hyps, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            code_emb = mean_pooling(side_model(**code_enc), code_enc["attention_mask"])
            hyp_emb = mean_pooling(side_model(**hyp_enc), hyp_enc["attention_mask"])
            sims = torch.sum(
                torch.nn.functional.normalize(code_emb, p=2, dim=1) *
                torch.nn.functional.normalize(hyp_emb, p=2, dim=1), dim=1).cpu().tolist()
    else:
        for code, hyp in zip(codes, hyps):
            code_emb = side_model.encode(code, convert_to_tensor=True, normalize_embeddings=True)
            hyp_emb = side_model.encode(hyp, convert_to_tensor=True, normalize_embeddings=True)
            sims.append(util.pytorch_cos_sim(code_emb, hyp_emb).item())
    return {"per_example": [round(float(s), 6) for s in sims]}
