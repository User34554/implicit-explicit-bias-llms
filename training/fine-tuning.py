import pandas as pd
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# 1. Load model + tokenizer
# -----------------------------
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# -----------------------------
# 2. LoRA config (smaller, safer)
# -----------------------------
peft_config = LoraConfig(
    r=4,                     # smaller rank
    lora_alpha=8,            # smaller scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,         # more dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# -----------------------------
# 3. Load + prepare dataset
# -----------------------------
df = pd.read_excel("bias_data.xlsx")
df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -----------------------------
# 4. Format into general template
# -----------------------------
def format_example(example):
    prompt = f"""Task: Decide whether the following sentence is Biased (1) or Unbiased (0).

Sentence:
{example['sentence']}

Label:"""
    completion = str(example["label"])  # "0" or "1"
    return {"prompt": prompt, "completion": completion}

dataset = dataset.map(format_example)

# -----------------------------
# 5. Tokenization
# -----------------------------
def tokenize(batch):
    # tokenize prompt + completion together
    tokenized = tokenizer(
        batch["prompt"],
        batch["completion"],
        truncation=True,
        padding="max_length",
        max_length=128,  # shorter for stability
    )

    # Mask out prompt tokens in labels
    labels = []
    for p, c in zip(batch["prompt"], batch["completion"]):
        prompt_ids = tokenizer(p, truncation=True, max_length=128)["input_ids"]
        completion_ids = tokenizer(c, truncation=True, max_length=10)["input_ids"]
        full_ids = prompt_ids + completion_ids

        if len(full_ids) < 128:
            full_ids += [tokenizer.pad_token_id] * (128 - len(full_ids))
        else:
            full_ids = full_ids[:128]

        # mask prompt part
        full_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)
        labels.append(full_ids)

    tokenized["labels"] = labels
    return tokenized

train_dataset = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(tokenize, remove_columns=dataset["test"].column_names)

# -----------------------------
# 6. Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    true_labels, pred_labels = [], []

    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != -100:
                true_labels.append(li)
                pred_labels.append(pi)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 7. Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="./llama3-bias-classifier",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,     # smaller learning rate
    num_train_epochs=3,     # enough but not too much
    fp16=False,
    bf16=True,
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -----------------------------
# 8. Save model
# -----------------------------
model.save_pretrained("llama3-bias-lora")
tokenizer.save_pretrained("llama3-bias-lora")

# -----------------------------
# 9. Inference example
# -----------------------------
prompt = """Task: Decide whether the following sentence is Biased (1) or Unbiased (0).

Sentence:
All people from X are lazy.

Label:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=2,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)
print("Predicted:", tokenizer.decode(outputs[0], skip_special_tokens=True))
