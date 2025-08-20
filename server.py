from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load GPT-NeoX-20B (requires large GPU or multiple GPUs)
MODEL_NAME = "EleutherAI/gpt-neox-20b"

print("Loading model... this may take a while.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"  # uses GPUs if available
)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}