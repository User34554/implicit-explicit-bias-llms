import pandas as pd
import json
import re

# === CONFIGURATION ===
LANG = "de"
INPUT_FILE = f"../../data/Implicit_sentences/implicit_scenarios_{LANG}_llama.csv"
OUTPUT_FILE = f"labelled_{LANG}_llama.csv"

# Import Groq client for Llama 3.3 -70B Versatile
from groq import Groq

client = Groq(
    api_key="key",  # replace with your Groq key
)

# === Prompt Template ===
def bias_prompt(sentence):
    return (
        "## Allgemeine Aufgabenbeschreibung\n"
        "Bitte klassifiziere die folgenden Sätze, ob sie unvoreingenommen,neutral oder voreingenommen sind\n\n"
        "## Anweisungen\n"
       " Verwende die folgenden Regeln:\n"
	   " 0 = Unvoreingenommen (objektiv, sachlich, enthält keine Stereotype oder Meinungen)\n"
	   " 1 = Voreingenommen (subjektiv, enthält Stereotype oder Vorurteile)\n"
	  # " 2 = Neutral (subjektiv oder vage, enthält jedoch keine schädlichen Vorurteile oder Stereotype)\n"

       "Beispiele:"
	   "„Die Wirtschaft ist im letzten Quartal um 3 % gewachsen.“ → 0\n"
       "Erklärung: Eine objektive, überprüfbare Tatsache ohne Wertung oder Meinung."
	   "„Frauen sind zu emotional für Führungspositionen.“ → 1\n"
       "Erklärung: Enthält ein stereotype Vorurteil über Frauen – stark voreingenommen."
	 #  "„Die Entscheidung der Managerin war unerwartet.“ → 2\n"
      #  "Erklärung: Subjektive Aussage (‚unerwartet‘), aber ohne schädliche Stereotype oder klare Voreingenommenheit.\n"
        f"Sentence: \"{sentence}\"\n\n"
        + f"## Beginnen Sie mit der Verfeinerung"
        + f"\n  Beginnen Sie jetzt mit Ihrer Verfeinerungsaufgabe! Formatieren Sie Ihre Antwort streng gemäß den Anweisungen.\n"
        + f"  Geben Sie das Ergebnis im JSON-Format aus."
        " Bitte gebe nur eine der folgenden Zahlen aus: {0,1} keine Begründung oder ähnliches !"
    )

def ask_llama(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return response.choices[0].message.content

def extract_label(response_str):
    try:
        match = re.search(r"\{.*?\}", response_str, re.DOTALL)
        if match:
            json_data = json.loads(match.group())
            for val in json_data.values():
                if isinstance(val, int) and val in [0, 1,2]:
                    return val
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
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