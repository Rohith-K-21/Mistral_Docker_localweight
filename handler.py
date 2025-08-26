import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path

MODEL_PATH = Path("/workspace/Mistral_weight")

print("Loading model...")

# Make sure the directory exists
if not MODEL_PATH.exists():
    raise ValueError(f"Model path does not exist: {MODEL_PATH}")

# Load tokenizer and model locally
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    use_fast=False  # prevents some HF auto checks
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True  # needed for some custom models like Mistral
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    framework="pt",
    device=0 if torch.cuda.is_available() else -1
)

def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt", "Hello")
    max_tokens = job_input.get("max_tokens", 128)
    temperature = job_input.get("temperature", 0.7)

    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,   
        temperature=temperature
    )

    return {"output": outputs[0]["generated_text"]}

# Start RunPod worker
runpod.serverless.start({"handler": handler})