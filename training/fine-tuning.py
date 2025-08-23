# =========================================
# LLaMA 3.1 8B LoRA Classification of Excel Bias Data
# =========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
import torch
import time
from transformers import TrainerCallback

# ----------------------------
# 0. Excel-File upload
# ----------------------------
excel_file = "bias_data.xlsx"  # <- mounted volume path
df = pd.read_excel(excel_file)
print("DataFrame loaded with shape:", df.shape)

# Ensure columns are named 'text' and 'label'
df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})

# ----------------------------
# 1. Train/Test Split
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.to_json("train.json", orient="records", lines=True)
test_df.to_json("test.json", orient="records", lines=True)
print("Train/Test split done.")

# ----------------------------
# 2. Hugging Face Dataset creation
# ----------------------------
dataset = DatasetDict({
    "train": load_dataset("json", data_files="train.json")["train"],
    "test": load_dataset("json", data_files="test.json")["train"]
})
print("Dataset created with train and test splits")

# ----------------------------
# 3. Set up tokenizer
# ----------------------------
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # important for LLaMA

# ----------------------------
# 4. Format dataset: input_text & target_text
# ----------------------------
def formatting_func(ex):
    return {
        "input_text": f"Label the following sentence as 0 (Unbiased) or 1 (Biased): {ex['text']}",
        "target_text": str(ex['label'])
    }

dataset["train"] = dataset["train"].map(formatting_func)
dataset["test"] = dataset["test"].map(formatting_func)
print(f"Dataset: {len(dataset['train'])} training examples, {len(dataset['test'])} test examples")

# ----------------------------
# 5. LoRA Configuration (with higher dropout & smaller rank for regularization)
# ----------------------------
lora_config = LoraConfig(
    r=32,                   # reduced rank
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # LLaMA-specific
    lora_dropout=0.1,       # increased dropout
    bias="none",
    task_type="CAUSAL_LM",
)

# ----------------------------
# 6. Load model (QLoRA / 4-bit)
# ----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

# ----------------------------
# 7. Training settings (with weight decay, smaller learning rate)
# ----------------------------
training_args = TrainingArguments(
    output_dir="llama8b-lora-bias",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,          # smaller learning rate
    weight_decay=0.01,            # weight decay for regularization
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",        # keep saving each epoch
    bf16=True
)

# ----------------------------
# 8. Step Timer Callback
# ----------------------------
class StepTimer(TrainerCallback):
    def __init__(self, warmup=10):
        self.warmup = warmup
        self.times = []
        self._t0 = None
        self.step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        dt = time.perf_counter() - self._t0
        self.step += 1
        if self.step > self.warmup:
            self.times.append(dt)
        if self.step % 10 == 0 and self.times:
            avg = sum(self.times[-10:]) / min(10, len(self.times))
            print(f"[Timing] avg step last 10: {avg:.4f}s")

# ----------------------------
# 9. SFT Trainer with Early Stopping
# ----------------------------
from transformers import EarlyStoppingCallback

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
)

#trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))  # stop if eval loss does not improve for 2 epochs

# ----------------------------
# 10. Training start
# ----------------------------
print("Starting training...")
trainer.train()
print("Training completed.")

# ----------------------------
# 11. Save model
# ----------------------------
model.save_pretrained("llama8b-lora-bias")
tokenizer.save_pretrained("llama8b-lora-bias")
print("Model and tokenizer saved to llama8b-lora-bias")

# ----------------------------
# 12. Load & test model
# ----------------------------
print("Loading model for testing...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, "llama8b-lora-bias")

# Inference
prompt = "Label the following sentence as 0 (Unbiased) or 1 (Biased): All people from X are lazy."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=1,         # generate only 1 token
    do_sample=False,          # deterministic output
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
print("Model output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Testing completed.")