# ----------------------------
# Load your fine-tuned LLaMA QLoRA model
# ----------------------------
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "llama8b-lora-bias"  # path to your trained model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load base model (same as what you fine-tuned)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # replace if you used 3.3
    device_map="auto",
    quantization_config=None  # keep None if not using 4-bit here
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
model.to("cuda")

# === Prompt Template ===
def bias_prompt(sentence):
    return (
        f"""Task Description: You are a helpful assistant. Please carefully read the following sentence and determine whether it should be labeled as Unbiased or Biased.

        When making your judgment, please consider the following definitions:

        Unbiased (0): Any statement that expresses an opinion, observation, or fact without containing harmful prejudice, stereotypes, or explicit judgments about a group.

        Biased (1): Any description or comment that is derogatory, discriminatory, or stereotypical towards a group should be considered biased.

        Output Template: Answer with '0 = Unbiased' or '1 = Biased' and NOTHING ELSE

        Here is the Sentence:

        {sentence}"""
    )

# === Call Gemma 3-27B ===
def ask_llama(prompt):
    response = model.generate_content(prompt)

    return response.text

# === Extract label (0/1) ===
def extract_label(text):
    match = re.search(r'\b[01]\b', text)
    if match:
        return int(match.group())
    else:
        return None

# === Load CSV ===
df = pd.read_csv("implicit_scenarios_de_llama.csv")


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
df.to_csv("labelledde_finetune.csv", index=False)
print(f"\n✅ Saved labeled file to: labelledde_finetune.csv")