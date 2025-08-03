import sys
from comet import download_model, load_from_checkpoint

def load_comet_model():
    comet_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(comet_path).eval()
    # print(" Loaded COMET model: wmt22-comet-da")
    return model
