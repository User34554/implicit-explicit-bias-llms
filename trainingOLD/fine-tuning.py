import pandas as pd
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

# ---- Speed/Determinism toggles (optional) ----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# 1) Load base model + tokenizer
# -----------------------------
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LEN = 256  # plenty for "Sentence + Label", adjust if you have very long sentences

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# safer with Trainer even if you later enable gradient checkpointing
model.config.use_cache = False

# -----------------------------
# 2) LoRA config
# -----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# -----------------------------
# 3) Load dataset from Excel
# -----------------------------
df = pd.read_excel("bias_data_en.xlsx")
# Force consistent column names: first col = sentence, second col = label
df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})
# Ensure labels are ints 0/1
df["label"] = df["label"].astype(int)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# -----------------------------
# 4) Custom Data Collator
#    - Tokenizes once per batch
#    - Masks everything before "Label:" so loss is only on the label token(s)
# -----------------------------

def _find_subsequence(seq: List[int], subseq: List[int]) -> Optional[int]:
    """Return start index of first subsequence occurrence, or None."""
    if not subseq or len(subseq) > len(seq):
        return None
    first = subseq[0]
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i] == first and seq[i : i + len(subseq)] == subseq:
            return i
    return None

@dataclass
class LabelAfterTemplateCollator:
    tokenizer: Any
    max_length: int = MAX_LEN
    template: str = "Label:"  # completion starts right after this template

    def __post_init__(self):
        self.template_ids = self.tokenizer(
            self.template, add_special_tokens=False
        )["input_ids"]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Build full texts INCLUDING the gold label so we can teacher-force and compute loss
        texts = [
            f"Sentence:\n{ex.get('sentence', ex.get('text', ''))}\nLabel: {int(ex['label'])}"
            for ex in examples
        ]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # Mask everything up to and including the "Label:" template tokens
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            start = _find_subsequence(ids, self.template_ids)
            if start is None:
                # If not found, ignore loss entirely for this sample
                labels[i] = torch.full_like(input_ids[i], -100)
            else:
                end = start + len(self.template_ids)
                labels[i, :end] = -100

        enc["labels"] = labels
        return enc

data_collator = LabelAfterTemplateCollator(tokenizer=tokenizer, max_length=MAX_LEN)

# -----------------------------
# 5) Metrics (token-level on unmasked positions = just the label token(s))
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    true_labels, pred_labels = [], []
    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != -100:
                true_labels.append(int(li))
                pred_labels.append(int(pi))

    if len(true_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 6) Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="./llama3-bias-classifier",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=20,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,  # << use our collator (no tokenizer arg -> no deprecation warning)
    # processing_class=tokenizer  # optional if you want to silence the future warning further
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# 7) Save LoRA adapter + tokenizer
# -----------------------------
model.save_pretrained("llama3-bias-lora")
tokenizer.save_pretrained("llama3-bias-lora")

# -----------------------------
# 8) Inference example (clean decode of ONLY new tokens)
# -----------------------------
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
base_model.config.use_cache = True
ft_model = PeftModel.from_pretrained(base_model, "llama3-bias-lora").eval()

test_sentence = "All people from X are lazy."
prompt = f"Sentence:\n{test_sentence}\nLabel:"
inputs = tokenizer(prompt, return_tensors="pt").to(ft_model.device)

with torch.no_grad():
    out = ft_model.generate(
        **inputs,
        max_new_tokens=2,   # room for "0"/"1" + maybe EOS
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

# Only decode the newly generated tokens (not the prompt)
new_tokens = out[0][inputs["input_ids"].shape[1]:]
pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print("Predicted label:", pred)