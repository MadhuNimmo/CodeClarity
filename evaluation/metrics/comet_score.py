import sys

def compute_comet_direct(codes, refs, hyps, comet_model):
    triplets = [{"src": str(src), "mt": str(hyp), "ref": str(ref)} for src, ref, hyp in zip(codes, refs, hyps)]
    try:
        pred = comet_model.predict(triplets, batch_size=8, gpus=1 if comet_model.hparams.gpus else 0)
        scores = [round(float(s), 6) if s is not None else None for s in pred.scores]
    except Exception as e:
        print(f"Error computing COMET: {e}")
        scores = [None] * len(triplets)

    return {"per_example": scores}
