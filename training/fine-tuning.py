import os
import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    __version__ as trf_version,
)
from packaging.version import parse as V
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 0. Basics / safety
# -----------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. Load model + tokenizer
# -----------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# -----------------------------
# 2. LoRA config (regularized)
# -----------------------------
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# -----------------------------
# 3. Load + prepare dataset
# -----------------------------
df = pd.read_excel("bias_data.xlsx")  # your uploaded file
# Expect: first col = sentence, second col = label
df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})
# Ensure labels are strings "0"/"1" (so completions tokenize as the correct token)
df["label"] = df["label"].astype(int).astype(str)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -----------------------------
# 4. Format into a general template
# -----------------------------
def format_example(example):
    prompt = (
        "Task: Decide whether the following sentence is Biased (1) or Unbiased (0).\n\n"
        "Sentence:\n"
        f"{example['sentence']}\n\n"
        "Label:"
    )
    completion = example["label"]  # "0" or "1"
    return {"prompt": prompt, "completion": completion}

dataset = dataset.map(format_example)

# -----------------------------
# 5. Tokenization (aligned & masked)
#    IMPORTANT: We concatenate prompt+completion BEFORE tokenization
# -----------------------------
MAX_LEN = 128

def tokenize(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]

    # Full texts are prompt + completion (no pair encoding)
    full_texts = [p + c for p, c in zip(prompts, completions)]
    enc = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    # Build labels: copy input_ids then mask the prompt part with -100
    labels = []
    for p, c, input_ids in zip(prompts, completions, enc["input_ids"]):
        # Re-tokenize prompt alone to know its length under same truncation
        prompt_ids = tokenizer(
            p, truncation=True, max_length=MAX_LEN
        )["input_ids"]

        # Create label array = input_ids, but mask prompt tokens
        lab = input_ids.copy()
        prompt_len = min(len(prompt_ids), MAX_LEN)

        # Mask prompt positions
        for i in range(prompt_len):
            lab[i] = -100

        # Also mask pads anywhere
        lab = [(-100 if tok == tokenizer.pad_token_id else tok) for tok in lab]
        labels.append(lab)

    enc["labels"] = labels
    return enc

train_dataset = dataset["train"].map(
    tokenize, remove_columns=dataset["train"].column_names
)
eval_dataset = dataset["test"].map(
    tokenize, remove_columns=dataset["test"].column_names
)

# -----------------------------
# 6. Metrics (robust: last non-masked token)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # logits: (bs, seq_len, vocab); labels: (bs, seq_len)
    # Preds are token IDs
    preds = logits.argmax(-1)

    true_labels = []
    pred_labels = []

    # For each sequence, pick the LAST non -100 label position
    for pseq, lseq in zip(preds, labels):
        # lseq is a 1D array of ints
        # Find indices where label is valid
        valid = (lseq != -100).nonzero()[0] if hasattr(lseq, "nonzero") else (lseq != -100).nonzero()
        # Handle both numpy & torch styles
        if hasattr(valid, "ndim") and valid.ndim > 1:
            valid = valid.squeeze()
        if len(valid) == 0:
            continue
        last_idx = valid[-1]
        true_labels.append(int(lseq[last_idx]))
        pred_labels.append(int(pseq[last_idx]))

    acc = accuracy_score(true_labels, pred_labels) if len(true_labels) else 0.0
    f1 = f1_score(true_labels, pred_labels, average="macro") if len(true_labels) else 0.0
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 7. Training args (version-safe)
# -----------------------------

training_args_kwargs = {
    "output_dir": "./llama3-bias-classifier",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "fp16": False,
    "bf16": torch.cuda.is_available(),  # use bf16 on capable GPUs
    "logging_steps": 20,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "report_to": "none",
}

training_args = TrainingArguments(**training_args_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------------
# 8. Save model
# -----------------------------
save_dir = "llama3-bias-lora"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# -----------------------------
# 9. Inference example (clean decode)
# -----------------------------
infer_prompt = (
    "Task: Decide whether the following sentence is Biased (1) or Unbiased (0).\n\n"
    "Sentence:\n"
    "All people from X are lazy.\n\n"
    "Label:"
)

inputs = tokenizer(infer_prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
gen_out = model.generate(
    **inputs,
    max_new_tokens=2,   # just need "0" or "1"
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
# Only decode the newly generated tokens:
gen_ids = gen_out[0][inputs["input_ids"].shape[1]:]
print("Predicted:", tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
