import pandas as pd
import json
import re

# === CONFIGURATION ===
LANG = "fr"
INPUT_FILE = f"implicit_scenarios_{LANG}_llama.csv"
OUTPUT_FILE = f"safe_guard eval.csv"

# Import Groq client for Llama 3.3 -70B Versatile
from groq import Groq

client = Groq(
    api_key="key",  # replace with your Groq key
)

# === Prompt Template ===

def ask_llama(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="meta-llama/llama-guard-4-12b",
    )
    return response.choices[0].message.content


# === Load CSV ===
df = pd.read_csv(INPUT_FILE)

# Check column positions
assert df.shape[1] >= 2, "Expected at least 2 columns"

# Initialize result list
evaluations = []

# Process each row in the DataFrame
for i, row in df.iterrows():
    implicit_sentence = row.iloc[1]  # second column
    print(f"\n🔹 Example {i + 1}: {implicit_sentence}")

    prompt = implicit_sentence
    response = ask_llama(prompt)
    print(f"-> Received response: {response}")

    match = re.search(r"(safe|unsafe|hate speech|self-harm|harassment)", response)
    label = match.group(0) if match else "Unknown"
    print(f"-> Label extracted: {label}")
    evaluations.append(label)

# Add evaluation as first column
df.insert(0, "evaluation", evaluations)

# Save result
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved labeled file to: {OUTPUT_FILE}")