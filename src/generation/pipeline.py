import json
from pathlib import Path
from tqdm import tqdm
from .datasets_loader import load_codesearchnet, load_code_rows_from_file
from .sampler import stratified_by_wordlen
from .prompts import PROMPTS, TARGET_NATURAL_LANGUAGES
from .model_registry import REGISTRY
from .generator import generate_summary

def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_pipeline(model_key: str,
                 model_id: str,
                 split: str,
                 out_dir: str,
                 samples_per_bucket: int = 3,
                 target_natural_languages=None,
                 source: str = "codesearchnet",
                 input_file: str = None):
    """
    flow:
    - Load CodeSearchNet (default) OR user-provided file of rows
    - Stratified sampling (short/medium/long) per programming language
    - For each prompt & target natural language → generate summary with chosen model
    - Save per-language JSON and combined JSON
    """
    # 1) Load data
    if source == "file":
        if not input_file:
            raise ValueError("source='file' requires --input_file path")
        df = load_code_rows_from_file(input_file)
    else:
        df = load_codesearchnet(split=split)

    # 2) Sample as in notebooks
    df_s = stratified_by_wordlen(df, samples_per_bucket=samples_per_bucket)

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 3) Model
    model_loader = REGISTRY[model_key]
    model, tok, device, apply_chat, gen_kwargs = model_loader(model_id=model_id)

    # 4) Prompts & languages
    target_langs = target_natural_languages or TARGET_NATURAL_LANGUAGES

    # 5) Generate
    for prompt_key, prompt_template in PROMPTS.items():
        all_results = []
        for prog_lang in df_s["language"].unique():
            lang_df = df_s[df_s["language"] == prog_lang]
            results = []

            for row in tqdm(lang_df.itertuples(index=False), total=len(lang_df), leave=False, desc=f"{prog_lang}-{prompt_key}"):
                entry = {
                    "id": f"sample_{getattr(row, 'index', 0)}",
                    "language": prog_lang,
                    "length_bucket": getattr(row, "length_bucket", "unknown"),
                    "word_len": getattr(row, "word_len", -1),
                    "code": row.whole_func_string,
                    "docstring": getattr(row, "func_documentation_string", ""),
                    "model_name": model_id,
                    "prompt_used": prompt_key
                }
                for nat_lang in target_langs:
                    entry[f"summary_{nat_lang.lower()}"] = generate_summary(
                        row.whole_func_string,
                        nat_lang,
                        prompt_template,
                        model, tok, device, apply_chat, gen_kwargs
                    )

                results.append(entry)
                all_results.append(entry)

            per_lang = out_root / f"{prog_lang.lower()}_{prompt_key}_summary_all_languages_{model_id.replace('/', '-')}.json"
            _write_json(per_lang, results)

        combined = out_root / f"all_languages_{prompt_key}_combined_{model_id.replace('/', '-')}.json"
        _write_json(combined, all_results)
