import torch
import evaluate
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- BERTScore ---
def compute_bertscore(refs, hyps):
    P, R, F1 = bert_score(hyps, refs, model_type="xlm-roberta-large", device=DEVICE, rescale_with_baseline=False)
    return {
        "precision": [round(p, 4) for p in P.cpu()],
        "recall": [round(r, 4) for r in R.cpu()],
        "f1": [round(f, 4) for f in F1.cpu()]
    }

# --- METEOR ---
def compute_meteor(refs, hyps):
    meteor = evaluate.load('meteor')
    return {
        "per_example": [round(meteor.compute(predictions=[h], references=[[r]])["meteor"], 6) for r, h in zip(refs, hyps)]
    }

# --- ROUGE-L ---
def compute_rougeL(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return {
        "per_example": [round(scorer.score(r, h)['rougeL'].fmeasure, 6) for r, h in zip(refs, hyps)]
    }

# --- BLEU ---
def compute_bleu(refs, hyps):
    return {
        "per_example": [round(sacrebleu.sentence_bleu(h, [r]).score / 100.0, 6) for r, h in zip(refs, hyps)]
    }

# --- ChrF ---
def compute_chrf(refs, hyps):
    return {
        "per_example": [round(sacrebleu.sentence_chrf(h, [r]).score / 100.0, 6) for r, h in zip(refs, hyps)]
    }
