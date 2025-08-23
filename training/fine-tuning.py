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
# 2. LoRA config
# -----------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# -----------------------------
# 3. Dataset preparation
# -----------------------------
df = pd.read_excel("bias_data.xlsx")
df = df.rename(columns={"text": "sentence", "label": "label"})

# Ensure labels are strings ("0" / "1")
df["label"] = df["label"].astype(str)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Format for fine-tuning
def format_example(example):
    prompt = f"Sentence:\n{example['sentence']}\nLabel:"
    completion = example["label"]  # "0" or "1"
    return {"prompt": prompt, "completion": completion}

dataset = dataset.map(format_example)

# -----------------------------
# 4. Tokenization
# -----------------------------
def tokenize(batch):
    # concatenate prompt + completion
    text = batch["prompt"] + " " + batch["completion"]
    tokenized = tokenizer(
        text, truncation=True, padding="max_length", max_length=256
    )

    # mask out loss for prompt tokens
    labels = tokenized["input_ids"].copy()
    prompt_len = len(tokenizer(batch["prompt"])["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels

    return tokenized

train_dataset = dataset["train"].map(tokenize, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(tokenize, remove_columns=dataset["test"].column_names)

# -----------------------------
# 5. Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    true_labels = []
    pred_labels = []
    for p, l in zip(preds, labels):
        for pi, li in zip(p, l):
            if li != -100:
                true_labels.append(li)
                pred_labels.append(pi)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 6. Training
# -----------------------------
training_args = TrainingArguments(
    output_dir="./llama3-bias-classifier",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,   # reduced LR to prevent overfitting
    weight_decay=0.01,    #  regularization
    num_train_epochs=5,   # more epochs but with early stopping
    fp16=False,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # needed for early stopping
    greater_is_better=True
)

# Add early stopping
from transformers import EarlyStoppingCallback
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks
)

trainer.train()