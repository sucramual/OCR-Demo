#!/usr/bin/env python3
"""
Minimal OCR Demo - MM1 Principle: Resolution > Connector
Demonstrates that increasing image resolution boosts OCR-style QA accuracy for multimodal LLMs
"""

import os
# Silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from pathlib import Path

import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import fitz  # PyMuPDF

from transformers import AutoProcessor, LlavaForConditionalGeneration

# Configuration
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
RESOLUTIONS = [224, 336, 448]  # Resolution sweep
CSV_PATH = Path("data/samples.csv")

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else (torch.bfloat16 if DEVICE == "mps" else torch.float32)

print(f"Device: {DEVICE} | dtype: {DTYPE}")
print(f"Loading model: {MODEL_ID}...")

# Load model and processor
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True
)

if DEVICE == "mps":
    model = model.to("mps")

print("Model loaded successfully!\n")


def load_image(path, target_res):
    """Load and resize image while maintaining aspect ratio. Supports PDFs and images."""
    path_str = str(path)

    # Handle PDF files using PyMuPDF
    if path_str.lower().endswith('.pdf'):
        # Open PDF and get first page
        pdf_doc = fitz.open(path_str)
        page = pdf_doc[0]  # First page

        # Render at high resolution (300 DPI equivalent)
        # PyMuPDF uses a matrix for scaling: 300 DPI = 4.17x scaling
        mat = fitz.Matrix(4.17, 4.17)
        pix = page.get_pixmap(matrix=mat)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pdf_doc.close()
    else:
        # Handle regular image files
        img = Image.open(path).convert("RGB")

    w, h = img.size

    # Resize so shorter side = target_res
    if w < h:
        new_w = target_res
        new_h = int(h * (target_res / w))
    else:
        new_h = target_res
        new_w = int(w * (target_res / h))

    img = img.resize((new_w, new_h), Image.BICUBIC)
    return img


def ask_llava(img, question, max_new_tokens=64, temperature=0.0):
    """Ask LLaVA a question about an image."""
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

    out = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Extract assistant response
    if "ASSISTANT:" in out:
        out = out.split("ASSISTANT:", 1)[-1].strip()

    return out.strip()


def normalize_text(s):
    """Normalize text for lenient exact-match comparison."""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # Strip punctuation for lenient matching
    s = re.sub(r"[^\w\s\.\-:/]", "", s)
    return s


def run_resolution_sweep():
    """Run inference across all resolutions and save results."""

    # Load dataset
    assert CSV_PATH.exists(), f"Error: {CSV_PATH} not found!"
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}\n")

    records = []

    for res in RESOLUTIONS:
        print(f"{'='*60}")
        print(f"Resolution: {res}px (shorter side)")
        print(f"{'='*60}")

        correct = 0
        total = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {res}px"):
            try:
                # Load image at target resolution
                img = load_image(row["image_path"], target_res=res)

                # Get prediction
                pred = ask_llava(img, str(row["question"]))
                gold = str(row["answer"])

                # Compute exact match
                pred_n = normalize_text(pred)
                gold_n = normalize_text(gold)
                em = int(pred_n == gold_n)

                records.append({
                    "image_path": row["image_path"],
                    "question": row["question"],
                    "gold": gold,
                    "pred": pred,
                    "em": em,
                    "resolution": res
                })

                correct += em
                total += 1

            except Exception as e:
                print(f"\nError processing {row['image_path']}: {e}")
                continue

        acc = correct / max(total, 1)
        print(f"\n✓ Resolution {res}px → EM accuracy: {acc:.3f} ({correct}/{total})\n")

    # Save results
    res_df = pd.DataFrame.from_records(records)
    res_df.to_csv("data/results_predictions.csv", index=False)
    print(f"✓ Saved: results_predictions.csv")

    # Aggregate by resolution
    res_summary = res_df.groupby("resolution")["em"].mean().reset_index()
    res_summary.to_csv("results_by_resolution.csv", index=False)
    print(f"✓ Saved: results_by_resolution.csv\n")

    return res_summary


def plot_results():
    """Generate accuracy vs resolution plot."""

    res_by = pd.read_csv("results_by_resolution.csv")

    plt.figure(figsize=(5, 3.2))
    plt.plot(res_by["resolution"], res_by["em"], marker="o", linewidth=2, markersize=8)
    plt.title("Exact-Match Accuracy vs Input Resolution (OCR-QA)", fontsize=12)
    plt.xlabel("Shortest Side Resolution (px)", fontsize=10)
    plt.ylabel("EM Accuracy", fontsize=10)
    plt.grid(True, linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig("resolution_vs_accuracy.png", dpi=180)
    print(f"✓ Saved: resolution_vs_accuracy.png")

    # Show summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(res_by.to_string(index=False))
    print("="*60)

    improvement = (res_by["em"].iloc[-1] - res_by["em"].iloc[0]) * 100
    print(f"\n📊 Resolution {RESOLUTIONS[0]}→{RESOLUTIONS[-1]} improved EM by {improvement:.1f} percentage points")
    print(f"📝 This aligns with Apple MM1's finding: Resolution > Connector tweaks\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MM1 OCR Demo: Resolution Scaling Experiment")
    print("="*60 + "\n")

    # Run the experiment
    res_summary = run_resolution_sweep()

    # Generate plot
    plot_results()

    print("\n✅ Demo complete! Check outputs:")
    print("   - results_predictions.csv (all predictions)")
    print("   - results_by_resolution.csv (accuracy by resolution)")
    print("   - resolution_vs_accuracy.png (visualization)")
