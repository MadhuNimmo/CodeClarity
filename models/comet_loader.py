import sys
from io import StringIO
from comet import download_model, load_from_checkpoint

def load_comet_model():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()

    comet_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(comet_path).eval()

    sys.stdout, sys.stderr = old_stdout, old_stderr
    print(" Loaded COMET model: wmt22-comet-da")
    return model
