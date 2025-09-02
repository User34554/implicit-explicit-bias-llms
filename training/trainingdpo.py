"""
Multi-Model DPO Training Script

Trains multiple models using Direct Preference Optimization on the same datasets
"""

import os

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

from trl import DPOTrainer, DPOConfig

import torch

# ==== Configuration ====

# base_dir = "/home/scserver/gdrive/fine-tune/Experiments/Regression/white_small/toxic" changed to current folder
base_dir = os.path.dirname(os.path.abspath(__file__))

# List of models to train with their specific target modules /only use the three
models_config = [
   # {
   #     "model_id": "cognitivecomputations/Wizard-Vicuna-7B-Uncensored",
   #     "target_modules": ["q_proj", "v_proj"]
  #  },
    {
        "model_id": "Orenguteng/Llama-3-8B-Lexi-Uncensored",
        "target_modules": ["q_proj", "v_proj"]
    },
    #{
    #    "model_id": "DavidAU/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B",
    #    "target_modules": ["q_proj", "v_proj"]
    #},
    #{
    #    "model_id": "mlabonne/gemma-3-4b-it-abliterated",
    #    "target_modules": ["q_proj", "v_proj"]
   # },
    #{
    #    "model_id": "xdrshjr/llama3.2_1b_uncensored_5000_8epoch_lora",
    #    "target_modules": ["q_proj", "v_proj"]
    #},
    #{
    #    "model_id": "heegyu/WizardVicuna-Uncensored-3B-0719",
   #     "target_modules": ["q_proj", "v_proj"]
   # },
   # {
    #    "model_id": "mlabonne/Qwen3-14B-abliterated",
   #     "target_modules": ["q_proj", "v_proj"]
   # },
    #{
   #     "model_id": "Fredithefish/Guanaco-3B-Uncensored",
   #     "target_modules": ["query_key_value", "dense"]
   # }
]

'''models_config = [
    {
        "model_id": "DavidAU/Deep-Reasoning-Llama-3.2-Instruct-uncensored-3B",
        "target_modules": ["q_proj", "v_proj"]
    },
    {
        "model_id": "mlabonne/gemma-3-4b-it-abliterated",
        "target_modules": ["q_proj", "v_proj"]
    },
    {
        "model_id": "xdrshjr/llama3.2_1b_uncensored_5000_8epoch_lora",
        "target_modules": ["q_proj", "v_proj"]
    },
    {
        "model_id": "mlabonne/Qwen3-14B-abliterated",
        "target_modules": ["q_proj", "v_proj"]
    },
]'''

# Get all train files (support both .json and .jsonl)
train_files = sorted(glob.glob(os.path.join(base_dir, "*_train.json*")))

# Process each dataset
for train_path in train_files:
    base_name = os.path.basename(train_path).split("_train.")[0]

    eval_path = os.path.join(base_dir, f"{base_name}_test.json")

    # Check for .jsonl if .json doesn't exist
    if not os.path.exists(eval_path):
        eval_path = os.path.join(base_dir, f"{base_name}_test.jsonl")

    if not os.path.exists(eval_path):
        print(f"Missing test file for {base_name}, skipping.")
        continue

    # Load datasets once per dataset (reuse for all models)
    print(f"📊 Loading dataset: {base_name}")

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_path, split="train")

    # Handle column name mapping for different formats
    if "accept" in train_dataset.column_names and "reject" in train_dataset.column_names:
        train_dataset = train_dataset.rename_columns({"accept": "chosen", "reject": "rejected"})

    if "accept" in eval_dataset.column_names and "reject" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_columns({"accept": "chosen", "reject": "rejected"})

    # Use small eval subset for efficiency
    eval_dataset = eval_dataset.select(range(10))

    # Train each model on this dataset
    for model_config in models_config:
        model_id = model_config["model_id"]
        target_modules = model_config["target_modules"]

        model_name = model_id.split("/")[-1]  # Extract model name for output directory
        output_dir = os.path.join(base_dir, f"{base_name}_{model_name}_dpo_ep3")

        print(f"🚀 Training {model_name} on {base_name} dataset...")
        print(f"   Target modules: {target_modules}")

        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, legacy=False, use_fast=True, trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                use_safetensors=True,
                attn_implementation="eager",
                trust_remote_code=True
            )

            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)

            # Configure LoRA with model-specific target modules
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=target_modules,  # Use model-specific target modules
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            model = get_peft_model(model, lora_config)

            # Training configuration
            training_args = DPOConfig(
                output_dir=output_dir,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=2,
                num_train_epochs=3,
                learning_rate=5e-5,
                gradient_checkpointing=True,
                # evaluation_strategy="epoch",
                logging_steps=10,
                save_strategy="epoch",
                bf16=True,
                remove_unused_columns=False,
                report_to="none"
            )

            training_args.logging_dir = os.path.join(output_dir, "logs")
            training_args.logging_strategy = "steps"

            # Reference model for KL divergence calculation
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True
            )

            # Initialize DPO trainer
            trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                ref_model=ref_model,
                processing_class=tokenizer,
            )

            # Start training
            trainer.train()

            # Save model and tokenizer
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            print(f"✅ Finished training {model_name} on {base_name} — model saved to {output_dir}")

            # Clean up GPU memory
            del model, ref_model, trainer, tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error training {model_name} on {base_name}: {str(e)}")

            # Clean up on error
            torch.cuda.empty_cache()
            continue

print("🎉 All training completed!")
