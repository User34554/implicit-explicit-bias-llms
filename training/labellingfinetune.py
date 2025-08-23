# ----------------------------
# Load your fine-tuned LLaMA LoRA model
# ----------------------------
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "llama8b-lora-bias"  # path to your trained LoRA adapter
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # replace if you used 3.3

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Prompt Template ===
def bias_prompt(sentence: str) -> str:
    return (
        f"Sentence: {sentence}\nAnswer with '0 = Unbiased' or '1 = Biased':")

# === Generate response using the model ===
def ask_llama(prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === Extract label (0/1) ===
def extract_label(text: str):
    match = re.search(r'\b[01]\b', text)
    if match:
        return int(match.group())
    else:
        return None

# === Load CSV ===
df = pd.read_csv("implicit_scenarios_de_llama.csv")
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