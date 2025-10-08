#!/usr/bin/env python3
"""Quick test to verify model loads correctly"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else (torch.bfloat16 if DEVICE == "mps" else torch.float32)

print(f"Device: {DEVICE} | dtype: {DTYPE}")
print(f"Loading model (this may take a few minutes on first run)...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)

if DEVICE == "mps":
    model = model.to("mps")

print("✅ Model loaded successfully!")
print(f"   Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
