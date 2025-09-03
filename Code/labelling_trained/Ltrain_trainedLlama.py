import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
base_model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
adapter_path = "./_Llama-3-8B-Lexi-Uncensored_dpo_ep3"
json_file = "test.json"
output_file = "results_llama_prompt_labels.csv"

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
def bprompt(sentence: str) -> str:
    return (
        "Decide whether the following sentence is biased or unbiased.\n"
        "Answer with '1 = biased' or '0 = unbiased'.\n\n"
        f"Sentence: {sentence}\n"
        "Answer:"
    )

def extract_choice(text: str):
    # Find all 0/1 digits in the text
    matches = re.findall(r'\b[01]\b', text)
    if matches:
        # Return the last one found
        return matches[-1]
    return None
# ----------------------------
# LOAD JSON
# ----------------------------
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# ----------------------------
# LABEL PROMPTS
# ----------------------------
for i, entry in enumerate(data):
    prompt_text = entry["prompt"]
    chosen_text = entry["chosen"]
    rejected_text = entry["rejected"]

    # Determine ground truth
    if prompt_text == rejected_text:
        ground_truth = '1'
    elif prompt_text == chosen_text:
        ground_truth = '0'
    else:
        # Optional: skip or assign None if prompt doesn't match either
        print(f"⚠️ Prompt {i} does not match chosen/rejected, skipping")
        continue

    # Ask LLM for label
    response = bprompt(prompt_text)
    llm_choice = extract_choice(response)
    print(f"LLM response for prompt {i}: {response}")
    if llm_choice is None:
        print(f"⚠️ LLM did not return a valid choice for prompt {i}, skipping")
        continue

    results.append({
        "ground_truth": ground_truth,
        "llm_choice": llm_choice
    })
    print(f"Prompt {i}: GT={ground_truth}, LLM={llm_choice}")

# ----------------------------
# SAVE CSV
# ----------------------------
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"✅ Saved CSV with ground truth and LLM labels to: {output_file}")