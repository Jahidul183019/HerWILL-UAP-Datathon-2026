"""XLM-R classification with inner checkpoint selection and a separate outer evaluation.

Standalone script for local MPS or Kaggle CUDA. Only competition CSVs are used.
Use a new run name for every experiment. Earlier submissions are never modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


def prepare_kaggle_environment():
    """Install the tested Python libraries while retaining Kaggle's CUDA PyTorch."""
    if not Path("/kaggle/working").is_dir():
        return
    versions = {"transformers": "5.9.0", "tokenizers": "0.22.2",
                "huggingface-hub": "1.17.0", "safetensors": "0.7.0",
                "numpy": "2.4.3", "pandas": "3.0.3", "scipy": "1.17.1",
                "scikit-learn": "1.8.0", "joblib": "1.5.3"}
    missing = []
    for package, wanted in versions.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != wanted:
            missing.append(f"{package}=={wanted}")
    if missing:
        require_restart = os.environ.get("SUBMISSION5_DEPENDENCIES_READY")
        if require_restart:
            raise RuntimeError(f"Dependency versions did not resolve: {missing}")
        print("Installing training dependencies:", missing, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
        os.environ["SUBMISSION5_DEPENDENCIES_READY"] = "1"
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv])


if __name__ == "__main__":
    prepare_kaggle_environment()

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


MODEL = "FacebookAI/xlm-roberta-base"
REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
LABELS = [0, 1, 2]
REFERENCE_F1 = 0.5376578864557929
EXPECTED_INPUT_SHA256 = {
    "train": "a269ad841f824c6e60930cfda8cb25bdce58a6a1ee01fc1fe663d23b19e34449",
    "test": "cca36f5416abfb8a41b1c84e5fa83aec18ffbff32fecd277c668a3f73312a38f",
    "sample_submission": "0f7d405dbd1e88ea285c7138514b7652f7441e995dcaed338bb208a86c2f347d",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def prune_unselected_weights(directory, selected_epoch):
    """Keep metrics for every epoch but retain model weights only for the selected one."""
    removed = []
    for checkpoint in sorted(directory.glob("epoch*")):
        if checkpoint.name == f"epoch{selected_epoch}":
            continue
        for pattern in ("*.safetensors", "pytorch_model*.bin"):
            for path in checkpoint.glob(pattern):
                removed.append(str(path.relative_to(directory)))
                path.unlink()
    retention = {
        "selected_checkpoint": f"epoch{selected_epoch}",
        "removed_weight_files": removed,
        "retained_for_all_epochs": ["configuration", "tokenizer", "validation logits when available"],
        "note": "The pinned initial model, source, settings and history reproduce every epoch.",
    }
    write_json(directory / "checkpoint_retention.json", retention)
    return retention


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def model_text(text):
    # XLM-R receives original casing; grouping still matches submissions 1–4.
    return " ".join(unicodedata.normalize("NFC", text).split())


def make_splits(train):
    groups = pd.factorize(train.text.map(normalize), sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit, holdout = next(outer.split(train.text, train.y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    a, b = next(inner.split(train.text.iloc[fit], train.y.iloc[fit], groups[fit]))
    require(not set(groups[fit]) & set(groups[holdout]), "Outer group overlap")
    require(not set(groups[fit[a]]) & set(groups[fit[b]]), "Inner group overlap")
    return fit[a], fit[b], fit, holdout


def load_data(data_dir):
    frames = []
    hashes = {}
    for name, columns in (("train", ["id", "text", "y"]),
                          ("test", ["id", "text"]),
                          ("sample_submission", ["id", "y_pred"])):
        path = data_dir / f"{name}.csv"
        frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        require(list(frame.columns) == columns, f"Unexpected {name} columns")
        require(frame.id.is_unique and frame.id.ne("").all(), f"Invalid {name} IDs")
        if "text" in frame:
            require(frame.text.map(model_text).ne("").all(), f"Empty {name} texts")
        frames.append(frame)
        hashes[name] = digest(path)
    train, test, sample = frames
    require(set(train.y.unique()) == set(LABELS), "Labels must be 0, 1, 2")
    require(not set(train.id) & set(test.id), "Train/test ID overlap")
    require(set(test.id) == set(sample.id), "Test/sample IDs differ")
    require(hashes == EXPECTED_INPUT_SHA256,
            "Competition inputs differ from the files used for submissions 1–4")
    return train, test, sample, hashes


def default_root():
    return Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(__file__).resolve().parents[1]


def find_data_dir(root):
    local = root / "data/raw"
    if all((local / f"{name}.csv").is_file() for name in EXPECTED_INPUT_SHA256):
        return local
    kaggle_input = Path("/kaggle/input")
    matches = [path.parent for path in kaggle_input.rglob("train.csv")
               if all((path.parent / f"{name}.csv").is_file() for name in EXPECTED_INPUT_SHA256)]
    require(len(matches) == 1, f"Expected one competition data directory; found {matches}")
    return matches[0]


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings, self.labels = encodings, labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def loader(texts, labels, tokenizer, args, shuffle=False):
    encoded = tokenizer([model_text(t) for t in texts], truncation=True,
                        max_length=args.max_length, padding=False)
    dataset = TextDataset(encoded, labels)

    def collate(rows):
        targets = [row.pop("labels") for row in rows] if "labels" in rows[0] else None
        batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
        if targets is not None:
            batch["labels"] = torch.tensor(targets, dtype=torch.long)
        return batch

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))


def metrics(y, predictions):
    return {"macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
            "classification_report": classification_report(y, predictions, labels=LABELS,
                                                             output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist()}


@torch.inference_mode()
def predict(model, batches, device):
    model.eval()
    scores = []
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**inputs).logits
        scores.append(logits.float().cpu().numpy())
    result = np.concatenate(scores)
    require(np.isfinite(result).all(), "Non-finite predictions")
    return result


def clear_device(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def new_model(args):
    set_seed(args.seed)
    source = args.pretrained or MODEL
    kw = {} if args.pretrained else {"revision": REVISION}
    return AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=3, attn_implementation="sdpa", **kw)


def train_phase(train, fit, val, epochs, directory, tokenizer, args, device):
    directory.mkdir()  # Refuse to overwrite any completed or interrupted phase.
    set_seed(args.seed)
    batches = loader(train.text.iloc[fit], train.y.iloc[fit].to_numpy(), tokenizer, args, True)
    validation = loader(train.text.iloc[val], None, tokenizer, args) if val is not None else None
    model = new_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=0.01, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    updates_per_epoch = math.ceil(len(batches) / args.accumulation)
    total_updates = updates_per_epoch * epochs
    warmup = max(1, round(total_updates * 0.1))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    counts = np.bincount(train.y.iloc[fit], minlength=3)
    weights = np.sqrt(len(fit) / (3 * counts))
    weights /= weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    history = []
    best_f1, best_epoch = -1.0, 1
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started, loss_sum = time.perf_counter(), 0.0
        for step, batch in enumerate(batches):
            y = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**inputs).logits
            # A fixed per-example weighted mean makes accumulation well-defined.
            losses = torch.nn.functional.cross_entropy(logits.float(), y, reduction="none")
            weighted_sum = (losses * weight_tensor[y]).sum()
            window_start = (step // args.accumulation) * args.accumulation * args.batch_size
            window_size = min(args.accumulation * args.batch_size, len(fit) - window_start)
            loss = weighted_sum / window_size
            require(bool(torch.isfinite(loss).item()), "Non-finite training loss")
            scaler.scale(loss).backward()
            loss_sum += float(weighted_sum.detach().cpu())
            if (step + 1) % args.accumulation == 0 or step + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"{directory.name} epoch {epoch}/{epochs}: {step+1}/{len(batches)} batches; "
                      f"{elapsed/(step+1):.2f}s/batch", flush=True)
        item = {"epoch": epoch, "training_loss": loss_sum / len(fit),
                "training_seconds": time.perf_counter() - started}
        checkpoint = directory / f"epoch{epoch}"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        if validation is not None:
            scores = predict(model, validation, device)
            item.update(metrics(train.y.iloc[val], scores.argmax(axis=1)))
            np.save(checkpoint / "validation_logits.npy", scores)
            if item["macro_f1"] > best_f1:
                best_f1, best_epoch = item["macro_f1"], epoch
        else:
            best_epoch = epoch
        history.append(item)
        write_json(directory / "history.json", history)
        print(json.dumps({k:v for k,v in item.items() if k not in ["classification_report", "confusion_matrix"]}), flush=True)
    result = {"selected_epoch": best_epoch, "class_weights": weights.tolist(), "history": history}
    write_json(directory / "complete.json", result)
    prune_unselected_weights(directory, best_epoch)
    del optimizer, model, scheduler
    clear_device(device)
    return result


def save_predictions(path, ids, y, scores):
    frame = pd.DataFrame({"id": list(ids), "y": list(y), "y_pred": scores.argmax(axis=1)})
    frame.to_csv(path, index=False)
    np.save(path.with_suffix(".logits.npy"), scores)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-name", default="submission5")
    parser.add_argument("--stage", choices=["all", "inner", "outer", "final"], default="all")
    parser.add_argument("--pretrained", help="Optional local copy of the pinned initial checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    require(min(args.epochs, args.batch_size, args.accumulation, args.max_length) > 0, "Invalid training settings")
    root = args.root.resolve()
    data_dir = args.data_dir or find_data_dir(root)
    run = root / "models" / args.run_name
    report = root / "reports" / f"{args.run_name}_metrics.json"
    output = root / "submissions" / f"{args.run_name}_xlmr.csv"
    val_output = root / "data/processed" / f"{args.run_name}_validation.csv"
    if args.stage in ["all", "inner"]:
        for path in [run, report, output, val_output]:
            require(not path.exists(), f"Preserving {path}; choose a new --run-name")
    else:
        require((run / "manifest.json").exists(), "Run inner stage first")
        require(not (run / args.stage).exists(), "Phase already exists; preserve it and use a new run")
        require(not output.exists(), "Submission already exists")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    require(device != "cpu", "Full XLM-R training needs a GPU; use MPS or a Kaggle GPU runtime")
    torch.set_num_threads(4)
    train, test, sample, hashes = load_data(data_dir)
    a, b, fit, holdout = make_splits(train)
    config = {k: getattr(args, k) for k in ["epochs", "batch_size", "accumulation", "max_length", "learning_rate", "seed"]}
    config.update(model=MODEL, revision=REVISION, loss="sqrt inverse frequency weighted cross entropy")
    if args.pretrained:
        initial_files = sorted(Path(args.pretrained).glob("*.safetensors"))
        require(bool(initial_files), "Local checkpoint must contain safetensors weights")
        config["initial_model_sha256"] = {p.name: digest(p) for p in initial_files}
    if args.stage in ["all", "inner"]:
        run.mkdir(parents=True)
        for directory in [report.parent, output.parent, val_output.parent]:
            directory.mkdir(parents=True, exist_ok=True)
        manifest = {"config": config, "input_sha256": hashes, "source_sha256": digest(__file__),
                    "device": device, "precision": "CUDA mixed float16" if device == "cuda" else "float32",
                    "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in
                        ["torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn"]},
                    "rows": {"inner_train":len(a), "inner_validation":len(b), "outer_train":len(fit),
                             "outer_validation":len(holdout), "train":len(train), "test":len(test)},
                    "reference_submission4_macro_f1":REFERENCE_F1,
                    "preprocessing":"NFC and whitespace; original case for XLM-R, lowercased grouping"}
        write_json(run / "manifest.json", manifest)
        shutil.copy2(__file__, run / "training_source.py")
        np.savez(run / "split_indices.npz", inner_train=a, inner_validation=b, outer_train=fit, outer_validation=holdout)
    else:
        manifest = json.loads((run / "manifest.json").read_text())
        require(manifest["config"] == config and manifest["input_sha256"] == hashes, "Run configuration/data changed")
        require(manifest["source_sha256"] == digest(__file__), "Training source changed since inner stage")
    kw = {} if args.pretrained else {"revision":REVISION}
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained or MODEL, **kw)
    print(f"Run {args.run_name}; device {device}; {manifest['rows']}", flush=True)
    if args.stage in ["all", "inner"]:
        train_phase(train, a, b, args.epochs, run / "inner", tokenizer, args, device)
    if args.stage == "inner":
        return
    selection = json.loads((run / "inner/complete.json").read_text())
    epochs = selection["selected_epoch"]
    if args.stage in ["all", "outer"]:
        train_phase(train, fit, None, epochs, run / "outer", tokenizer, args, device)
        checkpoint = run / "outer" / f"epoch{epochs}"
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
        scores = predict(model, loader(train.text.iloc[holdout], None, tokenizer, args), device)
        save_predictions(val_output, train.id.iloc[holdout], train.y.iloc[holdout], scores)
        result = metrics(train.y.iloc[holdout], scores.argmax(axis=1))
        result.update(manifest=manifest, selected_epoch=epochs, inner_history=selection["history"],
                      gain_over_submission4=result["macro_f1"] - REFERENCE_F1,
                      status="evaluated", limitation="Single reused outer holdout; not an independent final test.")
        write_json(report, result)
        del model
        clear_device(device)
        print(f"Outer Macro F1: {result['macro_f1']:.6f}; gain: {result['gain_over_submission4']:+.6f}", flush=True)
    result = json.loads(report.read_text())
    if args.stage == "outer":
        return
    if result["macro_f1"] <= REFERENCE_F1:
        print("No improvement over submission 4. Saved evaluation; no submission generated.", flush=True)
        return
    train_phase(train, np.arange(len(train)), None, epochs, run / "final", tokenizer, args, device)
    checkpoint = run / "final" / f"epoch{epochs}"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    batches = loader(test.text, None, tokenizer, args)
    scores = predict(model, batches, device)
    del model
    clear_device(device)
    reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    verified = predict(reloaded, batches, device)
    require(np.array_equal(scores.argmax(1), verified.argmax(1)), "Reload changed test predictions")
    lookup = pd.Series(scores.argmax(1), index=test.id)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(lookup)
    require(submission.y_pred.isin(LABELS).all(), "Invalid final labels")
    require(submission.id.equals(sample.id), "Submission IDs/order changed")
    require(not output.exists(), "Refusing to overwrite submission")
    submission.to_csv(output, index=False)
    reread = pd.read_csv(output, dtype={"id":str}, keep_default_na=False)
    require(reread.equals(submission), "Submission CSV round-trip mismatch")
    np.save(run / "test_logits.npy", scores)
    result.update(status="ready_for_manual_submission", submission_sha256=digest(output),
                  submission_rows=len(submission), final_model=str(checkpoint), reload_verified=True)
    write_json(report, result)
    print(f"Saved {output}; public score unknown.", flush=True)


if __name__ == "__main__":
    main()
