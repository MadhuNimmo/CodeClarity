import torch
from io import StringIO
import sys
import evaluate
import sacrebleu
from bert_score import score as bert_score
from rouge_score import rouge_scorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_bertscore(refs, hyps):
    if not refs or not hyps:
        return {"precision": [], "recall": [], "f1": []}
    P, R, F1 = bert_score(
        hyps, refs,
        model_type="xlm-roberta-large",
        lang="en",
        rescale_with_baseline=False,
        device=DEVICE
    )
    return {
        "precision": [round(p, 4) for p in P.cpu().tolist()],
        "recall": [round(r, 4) for r in R.cpu().tolist()],
        "f1": [round(f, 4) for f in F1.cpu().tolist()]
    }


def compute_meteor(refs, hyps):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    meteor = evaluate.load('meteor')

    sys.stdout = old_stdout
    sys.stderr = old_stderr

    per_example = []
    for r, h in zip(refs, hyps):
        try:
            k = meteor.compute(predictions=[h], references=[[r]])
            score = k["meteor"]
        except Exception:
            score = 0.0
        per_example.append(round(score, 6))
    return {"per_example": per_example}


def compute_chrf(refs, hyps):
    per_example = []
    for r, h in zip(refs, hyps):
        try:
            res = sacrebleu.sentence_chrf(h, [r], word_order=2)
            score = res.score / 100.0
        except Exception:
            score = 0.0
        per_example.append(round(score, 6))
    return {"per_example": per_example}


def compute_bleu(refs, hyps):
    per_example = []
    for r, h in zip(refs, hyps):
        try:
            res = sacrebleu.sentence_bleu(h, [r])
            score = res.score / 100.0
        except Exception:
            score = 0.0
        per_example.append(round(score, 6))
    return {"per_example": per_example}


def compute_rougeL(refs, hyps):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    per_example = []
    for r, h in zip(refs, hyps):
        try:
            sc = scorer.score(r, h)['rougeL'].fmeasure
        except Exception:
            sc = 0.0
        per_example.append(round(sc, 6))
    return {"per_example": per_example}
