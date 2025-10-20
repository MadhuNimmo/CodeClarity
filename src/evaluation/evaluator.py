import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from evaluation.metrics.metrics import compute_bertscore, compute_meteor, compute_chrf, compute_bleu, compute_rougeL
from evaluation.metrics.side_score import compute_side_score_batch
from evaluation.metrics.comet_score import compute_comet_direct
import torch

BACKTRANSLATION_LANGS = {
    "chinese": ("summary_chinese", "bt_chinese"),
    "french": ("summary_french", "bt_french"),
    "spanish": ("summary_spanish", "bt_spanish"),
    "portuguese": ("summary_portuguese", "bt_portuguese"),
    "arabic": ("summary_arabic", "bt_arabic"),
    "hindi": ("summary_hindi", "bt_hindi")
}


def count_total_samples(json_folder):
    total_samples = 0
    json_folder = Path(json_folder)
    json_files = sorted([p for p in json_folder.iterdir() if p.suffix == ".json"])

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            if data and isinstance(data, list):
                for lang_key, (summary_field, bt_field) in BACKTRANSLATION_LANGS.items():
                    if data and bt_field in data[0]:
                        bt_summaries = [d.get(bt_field, "").strip() for d in data]
                        if any(bt_summaries):
                            total_samples += len(data)
        except Exception:
            continue

    return total_samples

def evaluate_json_folder(json_folder, output_csv, side_model, side_tokenizer, comet_model, use_fallback_side):
    json_folder = Path(json_folder)
    json_files = sorted([p for p in json_folder.iterdir() if p.suffix == ".json"])
    total_samples = count_total_samples(json_folder)
    pbar = tqdm(total=total_samples, desc="Evaluating Samples")

    all_results = []
    for json_path in json_folder.glob("*.json"):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        refs = [d.get("summary_english", "").strip() for d in data]
        codes = [d.get("code", "") for d in data]
        model_name = data[0].get("model_name", "Unknown")
        prompt_used = data[0].get("prompt_used", "Unknown")

        for lang_key, (summary_field, bt_field) in BACKTRANSLATION_LANGS.items():
            if summary_field not in data[0] or bt_field not in data[0]:
                continue
            hyps_generated = [d.get(summary_field, "").strip() for d in data]
            hyps_bt = [d.get(bt_field, "").strip() for d in data]
            if not any(hyps_bt):
                continue

            # compute metrics
            bert = compute_bertscore(refs, hyps_bt)
            meteor = compute_meteor(refs, hyps_bt)
            chrf = compute_chrf(refs, hyps_bt)
            bleu = compute_bleu(refs, hyps_bt)
            rouge = compute_rougeL(refs, hyps_bt)
            comet = compute_comet_direct(codes, refs, hyps_bt, comet_model)
            side = compute_side_score_batch(
                codes, hyps_bt, side_model, side_tokenizer,
                "cuda" if torch.cuda.is_available() else "cpu", use_fallback_side
            )

            for i, entry in enumerate(data):
                result = {
                    "sample_id": entry.get("id", f"{i}"),
                    "code": entry.get("code", ""),
                    "docstring": entry.get("docstring", ""),
                    "programming_language": entry.get("language", ""),
                    "length_bucket": entry.get("length_bucket", ""),
                    "prompt_used": prompt_used,
                    "model_name": model_name,
                    "bt_language": lang_key,
                    "reference_summary": refs[i],
                    "generated_summary": hyps_generated[i],
                    "backtranslated_summary": hyps_bt[i],
                    "bertscore_f1": bert["f1"][i] if i < len(bert["f1"]) else None,
                    "bleu": bleu["per_example"][i] if i < len(bleu["per_example"]) else None,
                    "chrf++": chrf["per_example"][i] if i < len(chrf["per_example"]) else None,
                    "rougeL": rouge["per_example"][i] if i < len(rouge["per_example"]) else None,
                    "meteor": meteor["per_example"][i] if i < len(meteor["per_example"]) else None,
                    "comet": comet["per_example"][i] if i < len(comet["per_example"]) else None,
                    "side": side["per_example"][i] if i < len(side["per_example"]) else None,
                }
                all_results.append(result)
                pbar.update(1)

    pbar.close()
    print()
    if not all_results:
        print("ERROR: No valid results generated")
        return None
    df = pd.DataFrame(all_results)
    return df

