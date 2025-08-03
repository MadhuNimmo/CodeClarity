import sys
from io import StringIO

def compute_comet_direct(codes, refs, hyps, comet_model):
    triplets = [{"src": str(src), "mt": str(hyp), "ref": str(ref)} for src, ref, hyp in zip(codes, refs, hyps)]
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()

    try:
        pred = comet_model.predict(triplets, batch_size=8, gpus=1 if comet_model.hparams.gpus else 0)
        scores = [round(float(s), 6) if s is not None else None for s in pred.scores]
    except Exception as e:
        print(f"Error computing COMET: {e}")
        scores = [None] * len(triplets)

    sys.stdout, sys.stderr = old_stdout, old_stderr
    return {"per_example": scores}
