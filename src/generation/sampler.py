import pandas as pd

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure non-empty functions
    df = df[df["whole_func_string"].astype(str).str.strip().astype(bool)]
    df["word_len"] = df["whole_func_string"].apply(lambda x: len(str(x).split()))
    return df

def stratified_by_wordlen(df: pd.DataFrame, samples_per_bucket: int = 3, min_per_lang: int = 100) -> pd.DataFrame:
    """
    Same idea as your notebooks:
    - Per programming language
    - Filter, compute word_len
    - Split into short/medium/long thirds
    - Sample s/m/l (defaults 3/3/3)
    """
    langs = sorted(df["language"].dropna().unique().tolist())
    picks = []
    for lang in langs:
        dfl = _clean(df[df["language"] == lang])
        if len(dfl) < min_per_lang:
            # not enough examples in that language
            continue
        dfl = dfl.sort_values("word_len").reset_index(drop=True)
        n = len(dfl)
        short_df = dfl.iloc[: n // 3]
        med_df   = dfl.iloc[n // 3 : 2 * n // 3]
        long_df  = dfl.iloc[2 * n // 3 : ]

        try:
            s = short_df.sample(n=samples_per_bucket, random_state=42).copy()
            m = med_df.sample(n=samples_per_bucket, random_state=42).copy()
            l = long_df.sample(n=(9 - 2 * samples_per_bucket), random_state=42).copy()  # 3 buckets → 9 per lang
        except ValueError:
            continue

        s["length_bucket"], m["length_bucket"], l["length_bucket"] = "short", "medium", "long"
        picks.append(pd.concat([s, m, l], ignore_index=True))

    if not picks:
        raise RuntimeError("No samples collected. Check filtering or dataset availability.")
    return pd.concat(picks, ignore_index=True)
