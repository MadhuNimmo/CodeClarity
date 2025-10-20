import sys
import torch

def compute_comet_direct(codes, refs, hyps, comet_model):
    all_scores = []
    triplets = []

    for src, ref, hyp in zip(codes, refs, hyps):
        triplet = {
            "src": "" if src is None else str(src),
            "mt": "" if hyp is None else str(hyp),
            "ref": "" if ref is None else str(ref),
        }
        triplets.append(triplet)

    try:
        pred = comet_model.predict(triplets, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)

        # Extract Scores
        for idx, score in enumerate(pred.scores):
            if score is None:
                print(f"Triplet {idx}: SHIT.. None score")
            else:
                score = round(float(score), 6)
            all_scores.append(score)

    except Exception as e:
        print(f"Failed batch with error: {e}")
        all_scores.extend([None] * len(triplets))
    return {"per_example": all_scores}

def _extract_comet_scores(raw_pred):
    def _score_from_dict(d):
        if "system_score" in d and d["system_score"] is not None:
            return d["system_score"]
        if "scores" in d and isinstance(d["scores"], (list, tuple)) and d["scores"]:
            return d["scores"][0]
        return None

    out_scores = []
    if isinstance(raw_pred, list):
        for item in raw_pred:
            if hasattr(item, "to_dict"):
                d = item.to_dict()
            elif isinstance(item, dict):
                d = item
            else:
                try:
                    d = dict(item)
                except Exception:
                    d = {}
            out_scores.append(_score_from_dict(d))
    else:
        if hasattr(raw_pred, "to_dict"):
            d = raw_pred.to_dict()
        else:
            try:
                d = dict(raw_pred)
            except Exception:
                d = {}
        out_scores.append(_score_from_dict(d))

    normalized = []
    for s in out_scores:
        try:
            normalized.append(round(float(s), 6) if s is not None else None)
        except Exception:
            normalized.append(None)
    return normalized
