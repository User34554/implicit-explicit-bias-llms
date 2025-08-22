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

# ----------------------------
# 1. Train/Test Split
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print("Train/Test split done:")

# Optional: direkt JSONL speichern
train_df.to_json("train.json", orient="records", lines=True)
test_df.to_json("test.json", orient="records", lines=True)

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
tokenizer.pad_token = tokenizer.eos_token  # wichtig für LLaMA

# ----------------------------
# 4. Change Dataset into Prompt-Response format (pre-format for completion_only_loss)
# ----------------------------
def formatting_func(ex):
    return {"input_text": f"Sentence: {ex['text']}\nIs this sentence biased?\nAnswer: {'Yes' if ex['label'] == 1 else 'No'}"}

dataset["train"] = dataset["train"].map(formatting_func)
dataset["test"] = dataset["test"].map(formatting_func)
print(f"Dataset: {len(dataset['train'])} training examples, {len(dataset['test'])} test examples")

# ----------------------------
# 5. LoRA-Configuration
# ----------------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # LLaMA-spezifisch
    lora_dropout=0.05,
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
    device_map="auto"  # <- nutzt automatisch alle GPUs
)

# LoRA-Wrapper
model = get_peft_model(model, lora_config)
print(f"Model: {model}")

# Gradient Checkpointing + TF32
model.gradient_checkpointing_enable()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------
# 7. Training settings
# ----------------------------
training_args = TrainingArguments(
    output_dir="llama8b-lora-bias",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
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
# 9. SFT Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
   # completion_only_loss=True  # keep default
)
trainer.add_callback(StepTimer(warmup=10))

# ----------------------------
# 10. Training start
# ----------------------------
print("Starting training...")
trainer.train()
print("Training completed.")

# ----------------------------
# 11. Modell save
# ----------------------------
model.save_pretrained("llama8b-lora-bias")
tokenizer.save_pretrained("llama8b-lora-bias")
print("Model and tokenizer saved to llama8b-lora-bias")

# ----------------------------
# 12. Model load & test
# ----------------------------
print("Loading model for testing...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, "llama8b-lora-bias")

# Testing
prompt = "Sentence: 'All people from X are lazy.'\nIs this sentence biased?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=10)
print("Model output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Testing completed.")