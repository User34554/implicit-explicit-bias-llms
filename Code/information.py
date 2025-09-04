import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pprint import pprint

# ----------------------------
# CONFIG
# ----------------------------
base_model = "cognitivecomputations/Wizard-Vicuna-7B-Uncensored"  # base model from HF Hub
adapter_path = "./_Wizard-Vicuna-7B-Uncensored_dpo_ep3"           # folder with adapter_config.json

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base, adapter_path)  # wrap with LoRA
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model loaded!")

# ----------------------------
# BASE MODEL CONFIG
# ----------------------------
print("\n===== BASE MODEL CONFIG =====")
pprint(base.config.to_dict())

# ----------------------------
# GENERATION CONFIG
# ----------------------------
print("\n===== GENERATION CONFIG =====")
pprint(base.generation_config.to_dict())

# ----------------------------
# PEFT (ADAPTER) CONFIG
# ----------------------------
print("\n===== PEFT CONFIG =====")
if hasattr(model, "peft_config"):
    pprint(model.peft_config)
else:
    print("No PEFT config found (not a PEFT model).")

# ----------------------------
# PARAMETER COUNTS
# ----------------------------
print("\n===== PARAMETER COUNTS =====")
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable parameters: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.4f}%)")
print("\n===== training.args =====")
import torch

training_args = torch.load("/_Wizard-Vicuna-7B-Uncensored_dpo_ep3/checkpoint-144/training_args.bin", map_location="cpu")
print(training_args)
# ----------------------------
# SECOND LLM
# ----------------------------


# ----------------------------
# CONFIG
# ----------------------------
base_model = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
adapter_path = "./_Llama-3-8B-Lexi-Uncensored_dpo_ep3"       # folder with adapter_config.json

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base, adapter_path)  # wrap with LoRA
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("✅ Model loaded!")

# ----------------------------
# BASE MODEL CONFIG
# ----------------------------
print("\n===== BASE MODEL CONFIG =====")
pprint(base.config.to_dict())

# ----------------------------
# GENERATION CONFIG
# ----------------------------
print("\n===== GENERATION CONFIG =====")
pprint(base.generation_config.to_dict())

# ----------------------------
# PEFT (ADAPTER) CONFIG
# ----------------------------
print("\n===== PEFT CONFIG =====")
if hasattr(model, "peft_config"):
    pprint(model.peft_config)
else:
    print("No PEFT config found (not a PEFT model).")

# ----------------------------
# PARAMETER COUNTS
# ----------------------------
print("\n===== PARAMETER COUNTS =====")
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable parameters: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.4f}%)")

print("\n===== training.args =====")
import torch

training_args = torch.load("/_Llama-3-8B-Lexi-Uncensored_dpo_ep3/checkpoint-144/training_args.bin", map_location="cpu")
print(training_args)