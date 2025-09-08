#!/usr/bin/env python3
"""
Evaluate multilingual summaries with LitAI (LLM-as-a-judge).

- Reads a JSON ARRAY of samples in your provided schema.
- Scores six multilingual summaries (Chinese, French, Spanish, Portuguese, Arabic, Hindi).
- Deterministic judging by default: temperature=0.0
- Optional: include docstrings in the evaluation (default True, toggle with --no-docstring).
- Optional: --n-samples N picks up to N samples per programming language (field "language").
- Robust JSON extraction from model responses.
- --debug to dump raw responses or errors for failed parses.

Outputs:
  - per-sample scores JSON (compact, no comments)
  - aggregates JSON (means + counts per language)
"""

import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# LitAI client
from litai import LLM

MODEL_NAME_DEFAULT = "openai/gpt-5-nano"  # LitAI model id (change via --model)
TEMPERATURE_DEFAULT = 0.0
MAX_RETRIES_DEFAULT = 3
DEFAULT_THROTTLE_SEC = 0.05  # per-request throttle
DEFAULT_SLEEP_BETWEEN_LANG = 0.05  # extra sleep inside a sample

# Exactly six multilingual summaries (excluding English)
LANG_FIELDS = {
    "chinese": "summary_chinese",
    "french": "summary_french",
    "spanish": "summary_spanish",
    "portuguese": "summary_portuguese",
    "arabic": "summary_arabic",
    "hindi": "summary_hindi",
}

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
        sid = r.get("id") or r.get("sample_id") or str(r.get("_id") or "")
        samples.append(Sample(id=sid, code=r["code"], docstring=r.get("docstring"), raw=r))
    return samples

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _extract_json_block(text: str) -> str:
    """Extract best-effort JSON object from a text response (handles fences/prefix/suffix)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response text")
    # If code-fenced, slice the JSON portion
    if text.startswith("```"):
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last+1]
    # Try full text first
    try:
        json.loads(text)
        return text
    except Exception:
        # Fallback: find first {...} block heuristically
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last+1]
            # Quick sanity check
            json.loads(candidate)
            return candidate
        # Nothing usable
        raise ValueError("No JSON object found in response")

def _parse_json_strict(text: str) -> Dict[str, Any]:
    payload = _extract_json_block(text)
    return json.loads(payload)

def build_user_content(language: str, code: str, summary: str, docstring: Optional[str], use_docstring: bool) -> str:
    if use_docstring and docstring:
        return USER_TEMPLATE_WITH_DOC.format(language=language, code=code, doc=docstring, summary=summary)
    return USER_TEMPLATE_NO_DOC.format(language=language, code=code, summary=summary)

def _backoff_sleep(attempt: int, base: float = 0.75, cap: float = 8.0) -> None:
    # Exponential backoff with jitter
    delay = min(cap, base * (2 ** (attempt - 1)))
    delay = delay * (0.75 + 0.5 * (time.time() % 1.0))  # cheap jitter
    time.sleep(delay)

def _is_rate_limit_or_server_error(exc: Exception) -> bool:
    # Best-effort: LitAI may wrap HTTP errors; check message for 429/5xx hints.
    msg = str(exc).lower()
    return any(code in msg for code in ["429", "rate limit", "too many requests", "5xx", "500", "502", "503", "504"])

def litai_judge(
    model: LLM,
    language: str,
    code: str,
    summary: str,
    docstring: Optional[str],
    *,
    use_docstring: bool,
    temperature: float,
    max_retries: int,
    response_debug: bool = False,
    throttle_sec: float = 0.0,
) -> Dict[str, Any]:
    """Call LitAI model and return parsed JSON or a None-scored shell if summary missing."""
    if not summary or not summary.strip():
        return {
            "language": language,
            "scores": {d: None for d in ["correctness","completeness","clarity","terminology","brevity","overall"]}
        }

    user_text = build_user_content(language, code, summary, docstring, use_docstring)
    prompt = SYSTEM_PROMPT + "\n\n" + user_text

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if throttle_sec > 0:
                time.sleep(throttle_sec)
            resp = model.chat(prompt, temperature=temperature)  # LitAI returns text
            if response_debug:
                print(f"[DEBUG] raw response (len={len(str(resp))}): {str(resp)[:500]}")
            parsed = _parse_json_strict(resp)
            # Minimal schema guard
            parsed.setdefault("language", language)
            parsed.setdefault("scores", {})
            for k in ["correctness","completeness","clarity","terminology","brevity","overall"]:
                if k not in parsed["scores"]:
                    parsed["scores"][k] = None
            return parsed
        except Exception as e:
            last_exc = e
            if response_debug:
                print(f"[Retry {attempt}] Error: {e}")
            # Backoff more aggressively on rate-limit/server errors
            _backoff_sleep(attempt, base=1.0 if _is_rate_limit_or_server_error(e) else 0.5)

    raise RuntimeError(f"litai_judge failed after {max_retries} attempts: {last_exc}")

def evaluate_file(
    input_json: str,
    out_scores_json: str,
    out_aggregates_json: str,
    *,
    n_samples: Optional[int],
    sleep_between_lang: float,
    use_docstring: bool,
    temperature: float,
    max_retries: int,
    debug: bool,
    model_name: str,
    throttle_sec: float,
) -> None:
    api_key = os.getenv("LITAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set LITAI_API_KEY in your environment (or .env).")

    model = LLM(model=model_name, api_key=api_key)

    all_samples = load_samples(input_json)

    # Optional: take up to N samples per programming language
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
            try:
                judged = litai_judge(
                    model,
                    language=lang,
                    code=s.code,
                    summary=summary,
                    docstring=s.docstring,
                    use_docstring=use_docstring,
                    temperature=temperature,
                    max_retries=max_retries,
                    response_debug=debug,
                    throttle_sec=throttle_sec,
                )
                per_lang_scores[lang] = judged.get("scores", {})
            except Exception as e:
                if debug:
                    print(f"[ERROR] sample={s.id} lang={lang}: {e}")
                # Record a None-scored entry so one bad call doesn't kill the run
                per_lang_scores[lang] = {d: None for d in ["correctness","completeness","clarity","terminology","brevity","overall"]}
            if sleep_between_lang and sleep_between_lang > 0:
                time.sleep(sleep_between_lang)

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

    # Aggregates
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
        description="Evaluate multilingual summaries with LitAI (LLM-as-a-judge)."
    )
    parser.add_argument("input_json", help="Path to JSON file (array of sample objects).")
    parser.add_argument("--out-scores", default="llm_judgments.json", help="Output JSON with per-sample judgments.")
    parser.add_argument("--out-aggregates", default="llm_judgments_aggregates.json", help="Output JSON with aggregate means.")
    parser.add_argument("--no-docstring", action="store_true", help="Ignore docstring during evaluation.")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN_LANG, help="Seconds to sleep between per-language calls (inside a sample).")
    parser.add_argument("--throttle", type=float, default=DEFAULT_THROTTLE_SEC, help="Seconds to throttle before each API call (global pacing).")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of samples per programming language.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT, help="Sampling temperature (default 0.0).")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT, help="Max retries per judge call.")
    parser.add_argument("--debug", action="store_true", help="Print raw response text/errors when JSON parse fails.")
    parser.add_argument("--model", type=str, default=MODEL_NAME_DEFAULT, help="LitAI model id (e.g., openai/gpt-5-nano).")
    args = parser.parse_args()

    evaluate_file(
        input_json=args.input_json,
        out_scores_json=args.out_scores,
        out_aggregates_json=args.out_aggregates,
        n_samples=args.n_samples,
        sleep_between_lang=args.sleep,
        use_docstring=(not args.no_docstring),
        temperature=args.temperature,
        max_retries=args.max_retries,
        debug=args.debug,
        model_name=args.model,
        throttle_sec=args.throttle,
    )

if __name__ == "__main__":
    main()
