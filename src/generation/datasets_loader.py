
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

def load_codesearchnet(split: str = "test") -> pd.DataFrame:
    ds = load_dataset("code-search-net/code_search_net")
    return ds[split].to_pandas()

def load_code_rows_from_file(path: str) -> pd.DataFrame:
    """
    Optional: allow running the pipeline on a user-provided file containing code rows.
    Accepts JSON or JSONL with at least 'whole_func_string' and 'language' fields.
    """
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        rows = [json.loads(line) for line in p.open("r", encoding="utf-8")]
    elif p.suffix.lower() == ".json":
        rows = json.load(p.open("r", encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")
    if not isinstance(rows, list):
        raise ValueError("Input file must contain a list of records.")
    return pd.DataFrame(rows)
