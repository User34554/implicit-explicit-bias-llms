# =========================================
# LLaMA 3.1 8B LoRA Classification from CSV
# =========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
import torch
import time
from transformers import TrainerCallback

# ----------------------------
# 0. Load CSV
# ----------------------------
csv_file = "bias_data.csv"  # must have columns: sentence,label
df = pd.read_csv(csv_file)
print("CSV loaded with shape:", df.shape)

# ----------------------------
# 1. Train/Test split
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df.to_json("train.json", orient="records", lines=True)
test_df.to_json("test.json", orient="records", lines=True)

# ----------------------------
# 2. Create Hugging Face Dataset
# ----------------------------
dataset = DatasetDict({
    "train": Dataset.from_json("train.json"),
    "test": Dataset.from_json("test.json")
})

# ----------------------------
# 3. Tokenizer
# ----------------------------
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA needs this

# ----------------------------
# 4. Prompt-Response formatting
# ----------------------------
def format_prompt(ex):
    return {
        "input_text": f"Label this sentence as 0 (Unbiased) or 1 (Biased): {ex['sentence']}",
        "target_text": str(ex["label"])  # only numeric label
    }

dataset["train"] = dataset["train"].map(format_prompt)
dataset["test"] = dataset["test"].map(format_prompt)

# ----------------------------
# 5. LoRA config
# ----------------------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ----------------------------
# 6. Model + 4-bit QLoRA
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

# Wrap with LoRA
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------
# 7. Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="llama8b-lora-bias",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True
)

# ----------------------------
# 8. Optional Step Timer
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
# 9. Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    # completion_only_loss=True
)
trainer.add_callback(StepTimer(warmup=10))

# ----------------------------
# 10. Train
# ----------------------------
print("Starting training...")
trainer.train()
print("Training completed.")

# ----------------------------
# 11. Save model/tokenizer
# ----------------------------
model.save_pretrained("llama8b-lora-bias")
tokenizer.save_pretrained("llama8b-lora-bias")
print("Model saved.")

# ----------------------------
# 12. Test function
# ----------------------------
def ask_llama(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example
prompt = "Label this sentence as 0 (Unbiased) or 1 (Biased): 'All people from X are lazy.'"
print("Predicted label:", ask_llama(prompt))