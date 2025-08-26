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
# 2. LoRA config (smaller rank)
# -----------------------------
peft_config = LoraConfig(
    r=8,                     # reduced from 16
    lora_alpha=16,            # smaller learning scale
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# -----------------------------
# 3. Dataset preparation
# -----------------------------
df = pd.read_excel("bias_data_en.xlsx")
df = df.rename(columns={df.columns[0]: "sentence", df.columns[1]: "label"})

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -----------------------------
# 4. Format for fine-tuning
# -----------------------------
def format_example(example):
    prompt = f"Sentence:\n{example['sentence']}\nLabel:"  # only sentence
    completion = str(example["label"])                    # "0" or "1"
    return {"prompt": prompt, "completion": completion}

dataset = dataset.map(format_example)

# -----------------------------
# 5. Tokenization
# -----------------------------
def tokenize(batch):
    texts = [p + " " + c for p, c in zip(batch["prompt"], batch["completion"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    labels = []
    for i, (p, c) in enumerate(zip(batch["prompt"], batch["completion"])):
        # copy input_ids
        label_ids = tokenized["input_ids"][i][:]

        # find where prompt ends
        prompt_len = len(tokenizer(p, truncation=True, max_length=512)["input_ids"])

        # mask out prompt tokens
        label_ids[:prompt_len] = [-100] * prompt_len

        labels.append(label_ids)

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
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    logging_steps=20,
    report_to="none",
    save_total_limit=2,
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
prompt = "Sentence:\nAll people from X are lazy.\nLabel:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,   # only want the label (0 or 1)
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

# Slice off the prompt
generated = outputs[0][inputs["input_ids"].shape[1]:]
print("Predicted label:", tokenizer.decode(generated, skip_special_tokens=True))