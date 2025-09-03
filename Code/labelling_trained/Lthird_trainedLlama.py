# ----------------------------
# Load your fine-tuned LoRA model from a folder
# ----------------------------
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Paths
base_model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
adapter_path = "./_Llama-3-8B-Lexi-Uncensored_dpo_ep3"
excel_file = "dataset_third.xlsx"

# ----------------------------
# LOAD MODEL
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------------
# PROMPT & LABEL FUNCTIONS
# ----------------------------
def biased_or_unbiased_prompt(sentence: str) -> str:
    return (
        "Decide whether the following sentence is biased or unbiased.\n"
        "Answer with '1 = biased' or '0 = unbiased'.\n\n"
        f"Sentence: {sentence}\n"
        "Answer:"
    )

def ask_model(prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def extract_choice(text: str):
    # Find all 0/1 digits in the text
    matches = re.findall(r'\b[01]\b', text)
    if matches:
        # Return the last one found
        return matches[-1]
    return None

# ----------------------------
# LOAD EXCEL
# ----------------------------
df = pd.read_excel(excel_file)

results = []
correct = 0
total = 0

# ----------------------------
# EVALUATE SENTENCES
# ----------------------------
for _, row in df.iterrows():
    sentence = row["output"]
    human_label = row["both"]

    # Skip neutral labels
    if human_label == 0.5:
        continue

    prompt = biased_or_unbiased_prompt(sentence)
    response = ask_model(prompt)
    model_label = extract_choice(response)
    print(f"Choice: {model_label} | Response: {response}")
    is_correct = model_label == int(human_label)
    correct += int(is_correct)
    total += 1

    results.append({
        "sentence": sentence,
        "human_label": human_label,
        "model_label": model_label,
        "correct": is_correct
    })

    print(f"Processed {total}: model_label = {model_label}, correct? {is_correct}")

# ----------------------------
# COMPUTE ACCURACY
# ----------------------------
accuracy = correct / total * 100 if total > 0 else 0
print(f"\n✅ Accuracy on third dataset: {accuracy:.2f}%")

# ----------------------------
# SAVE RESULTS AS CSV
# ----------------------------
# Only keep the model's choice
choices_only = [res["model_label"] for res in results]

# Create a DataFrame
df_results = pd.DataFrame({"model_choice": choices_only})

# Save to CSV
output_file = "results_llama_third.csv"
df_results.to_csv(output_file, index=False)
print(f"✅ Saved CSV with model choices to: {output_file}")