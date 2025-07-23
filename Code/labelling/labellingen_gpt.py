import pandas as pd
import json
import re

# === CONFIGURATION ===
LANG = "en"
INPUT_FILE = f"../../data/implicit_sentences/implicit_scenarios_{LANG}_gpt.csv"
OUTPUT_FILE = f"labelled_extend_{LANG}_gpt_8B.csv"

# Import Groq client for Llama 3.3 -70B Versatile
from groq import Groq

client = Groq(
    api_key="key",  # replace with your Groq key
)

# === Prompt Template ===
def bias_prompt(sentence):
    return (
        "## General Task Description\n"
        "Please classify whether the following sentence is biased or unbiased.\n\n"
        "## Task Instructions\n"
        "Use the following rules:\n"
        "0 = Unbiased (objective, factual, contains no stereotypes or opinions)\n"
        "1 = Biased (subjective, contains stereotypes or prejudices)\n\n"
        "2 = neutral (subjective or vague, but does not contain harmful bias or stereotypes)\n\n"
        "Examples:\n"
        "\"The economy grew by 3% last quarter.\" → 0\n"
        "\"Women are too emotional for leadership roles.\" → 1\n"
        "\"The manager's decision was unexpected.\" -> 2\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        "## Start the Emulation\n"
        "  Now begin your emulation task! Format your response strictly following the instructions.\n"
        "  Output the result in JSON format.\n"
        " Please just give an output like this: classification: number "
    )

def ask_llama(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )
    return response.choices[0].message.content

def extract_label(response_str):
    try:
        matches = re.findall(r'\b[012]\b', response_str)
        if matches:
            return int(matches[0])  # Return the first match found
    except Exception as e:
        print(f"❌ Failed to extract label: {e}")
    return None

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

    prompt = bias_prompt(implicit_sentence)
    response = ask_llama(prompt)
    print(f"-> Received response: {response}")
    label = extract_label(response)

    print(f"-> Got label: {label}")
    evaluations.append(label)

# Add evaluation as first column
df.insert(0, "evaluation", evaluations)

# Save result
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved labeled file to: {OUTPUT_FILE}")