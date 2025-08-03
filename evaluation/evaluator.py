import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from metrics.metrics import compute_bertscore, compute_meteor, compute_chrf, compute_bleu, compute_rougeL
from metrics.side_score import compute_side_score_batch
from metrics.comet_score import compute_comet_direct

BACKTRANSLATION_LANGS = {
    "chinese": ("summary_chinese", "bt_chinese"),
    "french": ("summary_french", "bt_french"),
    "spanish": ("summary_spanish", "bt_spanish"),
    "portuguese": ("summary_portuguese", "bt_portuguese"),
    "arabic": ("summary_arabic", "bt_arabic"),
    "hindi": ("summary_hindi", "bt_hindi")
}
def count_total_samples(json_folder):
    total = 0
    json_folder = Path(json_folder)
    for json_path in json_folder.glob("*.json"):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for _, (_, bt_field) in BACKTRANSLATION_LANGS.items():
            if data and bt_field in data[0]:
                total += sum(1 for d in data if d.get(bt_field, "").strip())
    return total

def evaluate_json_folder(json_folder, output_csv, side_model, side_tokenizer, comet_model, use_fallback_side):
    json_folder = Path(json_folder)
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
            hyps_bt = [d.get(bt_field, "").strip() for d in data]
            if not any(hyps_bt):
                continue

            bert = compute_bertscore(refs, hyps_bt)
            meteor = compute_meteor(refs, hyps_bt)
            chrf = compute_chrf(refs, hyps_bt)
            bleu = compute_bleu(refs, hyps_bt)
            rouge = compute_rougeL(refs, hyps_bt)
            comet = compute_comet_direct(codes, refs, hyps_bt, comet_model)
            side = compute_side_score_batch(codes, hyps_bt, side_model, side_tokenizer, "cuda" if torch.cuda.is_available() else "cpu", use_fallback_side)

            for i, entry in enumerate(data):
                result = {
                    "sample_id": entry.get("id", f"{i}"),
                    "model_name": model_name,
                    "bt_language": lang_key,
                    "reference_summary": refs[i],
                    "backtranslated_summary": hyps_bt[i],
                    "bertscore_f1": bert["f1"][i],
                    "bleu": bleu["per_example"][i],
                    "chrf++": chrf["per_example"][i],
                    "rougeL": rouge["per_example"][i],
                    "meteor": meteor["per_example"][i],
                    "comet": comet["per_example"][i],
                    "side": side["per_example"][i],
                }
                all_results.append(result)
                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    return df
