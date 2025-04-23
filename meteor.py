import json
import pandas as pd
import time
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from googletrans import Translator
import nltk

nltk.download('punkt')

# === CONFIGURATION ===
languages = {
    "french": "codesearchnet_summary_french.json",
    "german": "codesearchnet_summary_german.json",
    "spanish": "codesearchnet_summary_spanish.json",
    "hindi": "codesearchnet_summary_hindi.json",
    "portuguese": "codesearchnet_summary_portuguese.json",
}
english_file = "codesearchnet_summary_english.json"
output_csv = "meteor_scores_only.csv"

translator = Translator()

# === HELPERS ===
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess(text):
    return word_tokenize(text.lower())

def compute_meteor(reference, hypothesis):
    return meteor_score([preprocess(reference)], preprocess(hypothesis))

async def back_translate(text, source_lang):
    try:
        # Translate the text to English using async
        back = await translator.translate(text, dest='en')
        return back.text
    except Exception as e:
        print(f"Translation error from {source_lang} to English: {e}")
        return ""

# === LOAD SUMMARIES ===
english_summaries = load_json(english_file)
language_summaries = {
    lang: load_json(file)
    for lang, file in languages.items()
}

# === BUILD RESULTS ===
def process_translations():
    results = []
    
    for i, english_obj in enumerate(english_summaries):
        english = english_obj["summary"]
        row = {"sample_id": i}
        
        for lang, lang_summaries in language_summaries.items():
            if i >= len(lang_summaries):
                print(f"Missing {lang} summary at index {i}")
                continue

            foreign_summary = lang_summaries[i]["summary"]
            back = back_translate(foreign_summary, lang)  # Asynchronous call for back translation
            time.sleep(1)  # Rate-limiting for translation API

            score = compute_meteor(english, back)
            row[f"meteor_{lang}"] = round(score, 4)

        results.append(row)

    # After gathering results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f" Saved METEOR scores to: {output_csv}")

# === EXECUTION ===
process_translations() 
