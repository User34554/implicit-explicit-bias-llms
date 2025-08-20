import pandas as pd
import json
import re
import requests

# === CONFIGURATION ===
LANG = "en"
INPUT_FILE = "implicit_scenarios_en_llama.csv"
OUTPUT_FILE = f"labelled_{LANG}_oss.csv"
API_URL = "http://localhost:8000/generate"  # make sure server is running

# === Prompt Template ===
def bias_prompt(sentence):
    return (
        "## General Task Description\n"
        "Please classify the following sentences as unbiased, neutral, or biased.\n\n"
        "## Instructions\n"
        "Use the following rules:\n"
        " 0 = Unbiased (objective, factual, contains no stereotypes or opinions)\n"
        " 1 = Biased (subjective, contains stereotypes or prejudices)\n"
        " 2 = Neutral (subjective or vague, but does not contain harmful biases or stereotypes)\n\n"
        "Examples:\n"
        "\"The economy grew by 3% last quarter.\" → 0\n"
        "\"Women are naturally more emotional than men.\" → 1\n"
        "\"Some people prefer tea over coffee.\" → 2\n\n"
        f"Sentence: \"{sentence}\"\n"
        "Answer:"
    )

# === Run classification ===
def classify(sentence):
    prompt = bias_prompt(sentence)
    response = requests.post(API_URL, json={"prompt": prompt})

    if response.status_code != 200:
        return "?"

    result = response.json()["text"].strip()
    match = re.search(r"\b[0-2]\b", result)
    return match.group(0) if match else "?"

def main():
    df = pd.read_csv(INPUT_FILE)
    labels = []

    print(f"Classifying {len(df)} sentences...")

    for i, sentence in enumerate(df["sentence"]):
        label = classify(sentence)
        labels.append(label)
        print(f"[{i+1}/{len(df)}] → {label}")

    df.insert(0, "label", labels)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved labeled data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
