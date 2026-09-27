"""Download XLM-R and measure a short training pass without creating a submission."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("No GPU available; use a GPU runtime for this benchmark")
    set_seed(42)
    torch.set_num_threads(4)
    model_id = "FacebookAI/xlm-roberta-base"
    revision = HfApi().model_info(model_id).sha
    print(f"Device: {device}; checkpoint: {model_id}@{revision}", flush=True)
    if device == "mps":
        print(subprocess.check_output(["sysctl", "-n", "hw.memsize", "hw.model"]).decode(), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, num_labels=3, attn_implementation="sdpa")
    tokenizer.save_pretrained(args.output / "pretrained")
    model.save_pretrained(args.output / "pretrained")
    model.to(device).train()
    train = pd.read_csv(root / "data/raw/train.csv").sample(n=200, random_state=42)
    batch = tokenizer(train.text.iloc[:args.batch_size].tolist(), truncation=True,
                      padding="max_length", max_length=args.max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = torch.tensor(train.y.iloc[:args.batch_size].tolist(), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
    def sync():
        torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
    times = []
    for step in range(6):
        sync()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Step {step + 1}: {elapsed:.2f}s; loss {float(loss.detach().cpu()):.4f}", flush=True)
    result = dict(device=device, model_id=model_id, revision=revision,
                  batch_size=args.batch_size, max_length=args.max_length, seconds=times,
                  estimated_inner_epoch_minutes=28690 / args.batch_size * sum(times[2:]) / 4 / 60,
                  note="Fixed-length full optimizer steps; actual dynamic padding and accumulation differ.")
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
