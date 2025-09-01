import pandas as pd
from groq import Groq

# === CONFIGURATION ===
INPUT_FILE = "bias_data_en.xlsx"
OUTPUT_FILE = "DPO_bias_data_en.xlsx"
MODEL_NAME = "llama-3.3-70b-versatile"

# Initialize Groq client

client = Groq(
    api_key="key",  # replace with your Groq key
)

# === Prompt Template ===
def debias_prompt(sentence):
    return f"""
You are an assistant that rewrites biased sentences into unbiased versions while preserving meaning and usefulness.
“Rewrite the following answer so it is unbiased, respectful, and free of stereotypes. Preserve the meaning and usefulness.”
In the output, write only the rewritten sentence and nothing else.
Here is an example, you should follow:

Input sentence: "In a prestigious university, a math professor, aware of a new student's background in a culture known for its rigorous educational system and high academic expectations, assumes the student will excel in an advanced calculus course without assessing their actual skills. The professor invites the student to join a highly competitive research project, citing their 'natural aptitude' for complex mathematical concepts, and expresses surprise when the student struggles to keep up with the coursework, eventually feeling pressured to drop the course due to the weight of unspoken expectations."
Output sentence: "At a prestigious university, a math professor meets a new student and, without first evaluating the student’s actual skills, assumes they will excel in an advanced calculus course. The professor invites the student to join a highly competitive research project, expecting strong performance. When the student struggles to keep up with the coursework, the professor is surprised, and the student eventually feels pressured to drop the course due to the weight of these unspoken expectations."

Original sentence:
"{sentence}"
"""

# === Call Groq LLM ===
def ask_groq(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# === Load Excel ===
df = pd.read_excel(INPUT_FILE)

# Ensure required columns exist
assert "sentence" in df.columns, "Expected a 'sentence' column"
assert "label" in df.columns, "Expected a 'label' column"

# Process rows
debiased_sentences = []
for i, row in df.iterrows():
    sentence = row["sentence"]
    label = row["label"]

    if label == 1:  # Only debias biased sentences
        prompt = debias_prompt(sentence)
        debiased = ask_groq(prompt)
        print(f"🔹 Debiased sentence {i + 1}: {debiased}")
    else:
        debiased = None  # keep empty for unbiased sentences

    debiased_sentences.append(debiased)

# Add new column
df["debiased_sentence"] = debiased_sentences

# Save to Excel
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Saved debiased dataset to: {OUTPUT_FILE}")