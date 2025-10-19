#!/usr/bin/env python3
"""
Pairwise LLM-as-a-Judge (Cohere ClientV2):
Compare direct non-English summaries vs pivoted (English -> target) summaries.

- Input: JSON ARRAY; each item has:
    code, docstring (optional), summary_<lang>, summary_english_to_<lang>
- For each of 6 target languages, compare DIRECT vs PIVOT under:
    correctness, completeness, clarity, terminology, brevity, overall
- Output 1: per-sample comparisons JSON
- Output 2: aggregates JSON (win/tie counts, mean scores)

Env:
  COHERE_API_KEY=<your key>
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# ---- Cohere v2 client ----
import cohere
from cohere import ClientV2

MODEL_NAME_DEFAULT = "command-a-03-2025"
TEMPERATURE_DEFAULT = 0.0
MAX_RETRIES_DEFAULT = 3
DEFAULT_SLEEP = 0.5

LANG_FIELDS = {
    "chinese": "summary_chinese",
    "french": "summary_french",
    "spanish": "summary_spanish",
    "portuguese": "summary_portuguese",
    "arabic": "summary_arabic",
    "hindi": "summary_hindi",
}

# System directive and strict JSON schema for pairwise judging
SYSTEM_PROMPT = (
    "You are an expert multilingual software documentation editor and code reviewer.\n"
    "You will receive CODE (and optional DOCSTRING) and two candidate SUMMARIES.\n"
    "Evaluate each candidate independently on the following dimensions (1-5 integers):\n"
    "  - correctness (technical accuracy)\n"
    "  - completeness (covers key behaviors/constraints)\n"
    "  - clarity (readability in target language)\n"
    "  - terminology (proper technical terms; identifiers preserved verbatim)\n"
    "  - brevity (concise without losing substance)\n"
    "  - overall (holistic quality)\n"
    "Then pick a winner per dimension: 'A', 'B', or 'Tie'.\n"
    "Return STRICT JSON only with this structure:\n"
    "{\n"
    '  "language": "<target>",\n'
    '  "scores": {\n'
    '    "A": {"correctness":1-5,"completeness":1-5,"clarity":1-5,"terminology":1-5,"brevity":1-5,"overall":1-5},\n'
    '    "B": {"correctness":1-5,"completeness":1-5,"clarity":1-5,"terminology":1-5,"brevity":1-5,"overall":1-5}\n'
    "  },\n"
    '  "winners": {"correctness":"A|B|Tie","completeness":"A|B|Tie","clarity":"A|B|Tie","terminology":"A|B|Tie","brevity":"A|B|Tie","overall":"A|B|Tie"},\n'
    '  "overall_winner": "A|B|Tie"\n'
    "}\n"
)

USER_TEMPLATE_WITH_DOC = (
    "TARGET_LANGUAGE: {language}\n\n"
    "CODE:\n{code}\n\n"
    "DOCSTRING:\n{doc}\n\n"
    "SUMMARY A (DIRECT {language}):\n{summary_a}\n\n"
    "SUMMARY B (PIVOT English→{language}):\n{summary_b}\n"
)

USER_TEMPLATE_NO_DOC = (
    "TARGET_LANGUAGE: {language}\n\n"
    "CODE:\n{code}\n\n"
    "SUMMARY A (DIRECT {language}):\n{summary_a}\n\n"
    "SUMMARY B (PIVOT English→{language}):\n{summary_b}\n"
)

DIMS = ["correctness", "completeness", "clarity", "terminology", "brevity", "overall"]

@dataclass
class Args:
    input_json: str
    out_comparisons: str
    out_aggregates: str
    temperature: float = TEMPERATURE_DEFAULT
    max_retries: int = MAX_RETRIES_DEFAULT
    sleep: float = DEFAULT_SLEEP
    n_samples: Optional[int] = None
    use_docstring: bool = True
    model: str = MODEL_NAME_DEFAULT
    debug: bool = False

def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _stringify_resp(resp: Any) -> str:
    text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    msg = getattr(resp, "message", None)
    if msg and hasattr(msg, "content"):
        parts = []
        for p in msg.content or []:
            t = getattr(p, "text", None) if not isinstance(p, dict) else p.get("text")
            if t:
                parts.append(t)
        if parts:
            return "".join(parts)
    try:
        return json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)))
    except Exception:
        return str(resp)

def _parse_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty response")
    if t.startswith("```"):
        s = t.find("{"); e = t.rfind("}")
        if s != -1 and e != -1 and e > s:
            t = t[s:e+1]
    try:
        return json.loads(t)
    except Exception:
        s = t.find("{"); e = t.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(t[s:e+1])
        raise

def _build_user(language: str, code: str, a: str, b: str, doc: Optional[str], use_doc: bool) -> str:
    tpl = USER_TEMPLATE_WITH_DOC if (use_doc and doc) else USER_TEMPLATE_NO_DOC
    return tpl.format(language=language, code=code, doc=(doc or ""), summary_a=a, summary_b=b)

def cohere_pairwise(
    client: ClientV2,
    *,
    model: str,
    language: str,
    code: str,
    summary_a: str,
    summary_b: str,
    docstring: Optional[str],
    use_docstring: bool,
    temperature: float,
    max_retries: int,
    debug: bool,
) -> Dict[str, Any]:
    user = _build_user(language, code, summary_a, summary_b, docstring, use_docstring)

    last_exc = None
    raw = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user",   "content": [{"type": "text", "text": user}]},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = _stringify_resp(resp)
            return _parse_json(raw)
        except Exception as e:
            last_exc = e
            time.sleep(0.4 * attempt)
    if debug and raw:
        print("[DEBUG raw]", raw[:2000])
    raise RuntimeError(f"pairwise judge failed after {max_retries} attempts: {last_exc}")

def evaluate(args: Args) -> None:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set COHERE_API_KEY.")

    client: ClientV2 = cohere.ClientV2(api_key)

    all_rows = load_samples(args.input_json)

    # Optional cap: up to N samples per programming language
    if args.n_samples is not None:
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for r in all_rows:
            pl = r.get("language") or "unknown"
            g = by_lang.setdefault(pl, [])
            if len(g) < args.n_samples:
                g.append(r)
        rows = [x for g in by_lang.values() for x in g]
    else:
        rows = all_rows

    print(f"Loaded {len(rows)} samples (after filtering).")

    comparisons: List[Dict[str, Any]] = []

    for r in tqdm(rows, desc="Comparing"):
        code = r.get("code", "")
        doc = r.get("docstring")
        sample_id = r.get("id")

        per_lang: Dict[str, Any] = {}
        for lang, direct_field in LANG_FIELDS.items():
            pivot_field = f"summary_english_to_{lang}"
            a = (r.get(direct_field) or "").strip()      # A = direct
            b = (r.get(pivot_field) or "").strip()       # B = pivot

            if not a and not b:
                per_lang[lang] = {"error": "both-empty"}
                continue

            try:
                out = cohere_pairwise(
                    client,
                    model=args.model,
                    language=lang,
                    code=code,
                    summary_a=a or "(empty)",
                    summary_b=b or "(empty)",
                    docstring=doc,
                    use_docstring=args.use_docstring,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    debug=args.debug,
                )
                per_lang[lang] = out
            except Exception as e:
                per_lang[lang] = {"error": str(e)}

            if args.sleep > 0:
                time.sleep(args.sleep)

        comparisons.append(
            {
                "id": sample_id,
                "comparisons": per_lang,
                "meta": {
                    "language": r.get("language"),
                    "length_bucket": r.get("length_bucket"),
                    "model_name": r.get("model_name"),
                },
            }
        )

    save_json(args.out_comparisons, comparisons)

    # ---- Aggregates ----
    agg = {
        lang: {
            "wins_A_direct": {d: 0 for d in DIMS},
            "wins_B_pivot":  {d: 0 for d in DIMS},
            "ties":          {d: 0 for d in DIMS},
            "mean_scores_A": {d: 0.0 for d in DIMS},
            "mean_scores_B": {d: 0.0 for d in DIMS},
            "count_scores":  {d: 0 for d in DIMS},
        }
        for lang in LANG_FIELDS.keys()
    }

    for row in comparisons:
        per_lang = row.get("comparisons", {})
        for lang, res in per_lang.items():
            if not isinstance(res, dict):
                continue
            winners = res.get("winners")
            scores = res.get("scores")
            if winners and scores:
                A = scores.get("A", {})
                B = scores.get("B", {})
                for d in DIMS:
                    w = winners.get(d)
                    if w == "A":
                        agg[lang]["wins_A_direct"][d] += 1
                    elif w == "B":
                        agg[lang]["wins_B_pivot"][d] += 1
                    elif w == "Tie":
                        agg[lang]["ties"][d] += 1
                    # accumulate means
                    a_val = A.get(d); b_val = B.get(d)
                    if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                        agg[lang]["mean_scores_A"][d] += float(a_val)
                        agg[lang]["mean_scores_B"][d] += float(b_val)
                        agg[lang]["count_scores"][d] += 1

    # finalize means
    for lang in agg.keys():
        for d in DIMS:
            c = agg[lang]["count_scores"][d]
            if c > 0:
                agg[lang]["mean_scores_A"][d] /= c
                agg[lang]["mean_scores_B"][d] /= c
            else:
                agg[lang]["mean_scores_A"][d] = None
                agg[lang]["mean_scores_B"][d] = None

    save_json(args.out_aggregates, {"dimensions": DIMS, "aggregates": agg})
    print(f"\nSaved comparisons to: {args.out_comparisons}")
    print(f"Saved aggregates to:   {args.out_aggregates}")

def main():
    p = argparse.ArgumentParser(description="Pairwise judge: direct vs pivot summaries (Cohere ClientV2)")
    p.add_argument("input_json", help="Path to input JSON array")
    p.add_argument("--out-comparisons", default="pairwise_comparisons.json", help="Per-sample output JSON")
    p.add_argument("--out-aggregates",  default="pairwise_aggregates.json", help="Aggregates output JSON")
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES_DEFAULT)
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    p.add_argument("--n-samples", type=int, default=None, help="Max samples per programming language (input 'language' field)")
    p.add_argument("--no-docstring", action="store_true", help="Ignore docstring during evaluation")
    p.add_argument("--model", type=str, default=MODEL_NAME_DEFAULT)
    p.add_argument("--debug", action="store_true")
    args_ns = p.parse_args()

    args = Args(
        input_json=args_ns.input_json,
        out_comparisons=args_ns.out_comparisons,
        out_aggregates=args_ns.out_aggregates,
        temperature=args_ns.temperature,
        max_retries=args_ns.max_retries,
        sleep=args_ns.sleep,
        n_samples=args_ns.n_samples,
        use_docstring=(not args_ns.no_docstring),
        model=args_ns.model,
        debug=args_ns.debug,
    )
    evaluate(args)

if __name__ == "__main__":
    main()
