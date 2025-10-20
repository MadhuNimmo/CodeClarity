#!/usr/bin/env python3
"""
Evaluate multilingual summaries with Cohere command-a (LLM-as-a-judge) using ClientV2.

- Reads a JSON ARRAY of samples in your provided schema.
- Scores six multilingual summaries (Chinese, French, Spanish, Portuguese, Arabic, Hindi).
- Deterministic judging: temperature=0.0
- Optional: include docstrings in the evaluation (default True, toggle with --no-docstring).
- Optional: --n-samples N picks up to N samples per programming language (field "language").
- Robust JSON extraction from Cohere v2 responses, with response_format forcing JSON.
- --debug to dump raw responses for failed parses.

Outputs:
  - per-sample scores JSON (compact, no comments)
  - aggregates JSON (means + counts per language)
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- Cohere v2 client ----
import cohere
from cohere import ClientV2

MODEL_NAME_DEFAULT = "command-a-03-2025"
TEMPERATURE_DEFAULT = 0.0
MAX_RETRIES_DEFAULT = 3

# Exactly six multilingual summaries (excluding English)
LANG_FIELDS = {
    "chinese": "summary_chinese",
    "french": "summary_french",
    "spanish": "summary_spanish",
    "portuguese": "summary_portuguese",
    "arabic": "summary_arabic",
    "hindi": "summary_hindi",
}

# Minimal system directive + terse output (no comments) to save tokens
SYSTEM_PROMPT = (
    "You are an expert multilingual software documentation editor and code reviewer.\n"
    "Evaluate the SUMMARY in TARGET_LANGUAGE against CODE and optional DOCSTRING.\n"
    "Score using integers 1-5 for: correctness, completeness, clarity, terminology, brevity, and overall.\n"
    "Return STRICT JSON with only keys:\n"
    '{"language":"<name>","scores":{"correctness":1-5,"completeness":1-5,"clarity":1-5,'
    '"terminology":1-5,"brevity":1-5,"overall":1-5}}\n'
    "Do not include any additional text."
)

USER_TEMPLATE_WITH_DOC = (
    "TARGET_LANGUAGE: {language}\n\n"
    "CODE:\n{code}\n\n"
    "DOCSTRING:\n{doc}\n\n"
    "SUMMARY:\n{summary}\n"
)

USER_TEMPLATE_NO_DOC = (
    "TARGET_LANGUAGE: {language}\n\n"
    "CODE:\n{code}\n\n"
    "SUMMARY:\n{summary}\n"
)

@dataclass
class Sample:
    id: str
    code: str
    docstring: Optional[str]
    raw: Dict[str, Any]

def load_samples(path: str) -> List[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    samples: List[Sample] = []
    for r in rows:
        sid = r.get("id") or r.get("sample_id")
        samples.append(Sample(id=sid, code=r["code"], docstring=r.get("docstring"), raw=r))
    return samples

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _stringify_resp(resp: Any) -> str:
    """
    Extract text from a Cohere v2 chat response across possible fields.
    """
    # 1) Some SDKs expose `output_text`
    text = getattr(resp, "output_text", None)
    if text:
        return text if isinstance(text, str) else str(text)

    # 2) Classic `text`
    text = getattr(resp, "text", None)
    if text:
        return text if isinstance(text, str) else str(text)

    # 3) message.content list of {"type": "text", "text": "..."}
    msg = getattr(resp, "message", None)
    if msg and hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, list):
            parts = []
            for p in content:
                # p may be a dict or an SDK object with .type/.text
                t = None
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        t = p.get("text")
                else:
                    # SDK object
                    if getattr(p, "type", None) == "text":
                        t = getattr(p, "text", None)
                if t:
                    parts.append(t)
            if parts:
                return "".join(parts)

    # 4) fallback: try to_dict / jsonable
    try:
        return json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)))
    except Exception:
        return str(resp)

def _parse_json_strict(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response text")

    # Remove markdown fences if present
    if text.startswith("```"):
        # grab first fenced block
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first:last+1]

    # Try strict parse
    try:
        return json.loads(text)
    except Exception:
        # Heuristic: find first {...}
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
        raise

def build_user_content(language: str, code: str, summary: str, docstring: Optional[str], use_docstring: bool) -> str:
    if use_docstring and docstring:
        return USER_TEMPLATE_WITH_DOC.format(language=language, code=code, doc=docstring, summary=summary)
    return USER_TEMPLATE_NO_DOC.format(language=language, code=code, summary=summary)

def cohere_judge(
    client: ClientV2,
    language: str,
    code: str,
    summary: str,
    docstring: Optional[str],
    *,
    use_docstring: bool,
    temperature: float,
    max_retries: int,
    response_debug: bool = False,
) -> Dict[str, Any]:
    """
    Calls Cohere chat (ClientV2) with response_format forcing JSON and returns parsed dict.
    """
    if not summary or not summary.strip():
        return {
            "language": language,
            "scores": {
                "correctness": None,
                "completeness": None,
                "clarity": None,
                "terminology": None,
                "brevity": None,
                "overall": None,
            },
        }

    user_text = build_user_content(language, code, summary, docstring, use_docstring)

    last_exc = None
    raw_text_snapshot = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(
                model=MODEL_NAME_DEFAULT,  # keep model stable unless overridden elsewhere
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",   "content": [{"type": "text", "text": user_text}]},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},  # <-- force JSON
                # You can add max_output_tokens if you want: max_output_tokens=256,
            )

            text = _stringify_resp(resp)
            raw_text_snapshot = text
            data = _parse_json_strict(text)
            return data
        except Exception as e:
            last_exc = e
            time.sleep(0.4 * attempt)

    if response_debug and raw_text_snapshot is not None:
        print("\n---- DEBUG: Raw response text that failed to parse ----")
        print(raw_text_snapshot[:2000])
        print("---- END DEBUG ----\n")

    raise RuntimeError(f"Cohere judging failed after {max_retries} attempts: {last_exc}")

def evaluate_file(
    input_json: str,
    out_scores_json: str,
    out_aggregates_json: str,
    *,
    n_samples: Optional[int],
    sleep: float,
    use_docstring: bool,
    temperature: float,
    max_retries: int,
    debug: bool,
) -> None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set COHERE_API_KEY in your environment (or .env).")

    client: ClientV2 = cohere.ClientV2(api_key)

    all_samples = load_samples(input_json)

    # Optional: take up to N samples per programming language (field "language")
    if n_samples is not None:
        by_pl: Dict[str, List[Sample]] = {}
        for s in all_samples:
            pl = s.raw.get("language") or "unknown"
            group = by_pl.setdefault(pl, [])
            if len(group) < n_samples:
                group.append(s)
        samples = [s for group in by_pl.values() for s in group]
    else:
        samples = all_samples

    print(f"Loaded {len(samples)} samples (after filtering).")

    results_per_sample: List[Dict[str, Any]] = []

    for s in tqdm(samples, desc="Evaluating"):
        per_lang_scores: Dict[str, Dict[str, Optional[int]]] = {}

        for lang, field in LANG_FIELDS.items():
            summary = s.raw.get(field, "")
            judged = cohere_judge(
                client,
                language=lang,
                code=s.code,
                summary=summary,
                docstring=s.docstring,
                use_docstring=use_docstring,
                temperature=temperature,
                max_retries=max_retries,
                response_debug=debug,
            )
            per_lang_scores[lang] = judged.get("scores", {})
            if sleep and sleep > 0:
                time.sleep(sleep)

        results_per_sample.append(
            {
                "id": s.id,
                "judgments": per_lang_scores,
                "meta": {
                    "language": s.raw.get("language"),
                    "length_bucket": s.raw.get("length_bucket"),
                    "model_name": s.raw.get("model_name"),
                },
            }
        )

    save_json(out_scores_json, results_per_sample)

    # Aggregates (means)
    dims = ["correctness", "completeness", "clarity", "terminology", "brevity", "overall"]
    agg: Dict[str, Dict[str, float]] = {lang: {d: 0.0 for d in dims} for lang in LANG_FIELDS.keys()}
    counts: Dict[str, Dict[str, int]] = {lang: {d: 0 for d in dims} for lang in LANG_FIELDS.keys()}

    for row in results_per_sample:
        j = row["judgments"]
        for lang, scores in j.items():
            for d in dims:
                val = scores.get(d)
                if isinstance(val, (int, float)):
                    agg[lang][d] += float(val)
                    counts[lang][d] += 1

    means = {
        lang: {d: (agg[lang][d] / counts[lang][d]) if counts[lang][d] else None for d in dims}
        for lang in LANG_FIELDS.keys()
    }

    aggregates_payload = {
        "dimensions": dims,
        "means_by_language": means,
        "counts_by_language": counts,
        "n_samples": len(results_per_sample),
    }
    save_json(out_aggregates_json, aggregates_payload)

    print(f"\nSaved per-sample scores to: {out_scores_json}")
    print(f"Saved aggregates to: {out_aggregates_json}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual summaries with Cohere command-a (ClientV2) as LLM-as-a-judge."
    )
    parser.add_argument("input_json", help="Path to JSON file (array of sample objects).")
    parser.add_argument("--out-scores", default="llm_judgments.json", help="Output JSON with per-sample judgments.")
    parser.add_argument("--out-aggregates", default="llm_judgments_aggregates.json", help="Output JSON with aggregate means.")
    parser.add_argument("--no-docstring", action="store_true", help="Ignore docstring during evaluation.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples per programming language.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT, help="Sampling temperature (default 0.0).")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT, help="Max retries per judge call.")
    parser.add_argument("--debug", action="store_true", help="Print raw response text when JSON parse fails.")
    args = parser.parse_args()

    evaluate_file(
        input_json=args.input_json,
        out_scores_json=args.out_scores,
        out_aggregates_json=args.out_aggregates,
        n_samples=args.n_samples,
        sleep=args.sleep,
        use_docstring=(not args.no_docstring),
        temperature=args.temperature,
        max_retries=args.max_retries,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
