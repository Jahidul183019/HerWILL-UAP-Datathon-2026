"""XLM-R-large two-seed training with nested grouped validation.

Standalone private Kaggle CUDA script; local execution is limited to CPU tests. Only competition CSVs are used.
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
        require_restart = os.environ.get("SUBMISSION6_DEPENDENCIES_READY")
        if require_restart:
            raise RuntimeError(f"Dependency versions did not resolve: {missing}")
        print("Installing training dependencies:", missing, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
        os.environ["SUBMISSION6_DEPENDENCIES_READY"] = "1"
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv])


if __name__ == "__main__":
    if Path('/kaggle/working').is_dir():
        # Cache the 2 GB initial checkpoint outside the collected output tree.
        os.environ['HF_HOME'] = '/tmp/submission6_hf'
        os.environ['HF_HUB_CACHE'] = '/tmp/submission6_hf/hub'
    prepare_kaggle_environment()

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


MODEL = "FacebookAI/xlm-roberta-large"
REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
LABELS = [0, 1, 2]
REFERENCE_F1 = 0.5590271875463054
SEEDS = (42, 2026)
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


def make_splits(train, fold=0):
    groups = pd.factorize(train.text.map(normalize), sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit, holdout = list(outer.split(train.text, train.y, groups))[fold]
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
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=3, attn_implementation="sdpa", **kw)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def train_phase(train, fit, val, epochs, directory, tokenizer, args, device, keep_all=False):
    directory.mkdir()  # Refuse to overwrite any completed or interrupted phase.
    set_seed(args.seed)
    batches = loader(train.text.iloc[fit], train.y.iloc[fit].to_numpy(), tokenizer, args, True)
    validation = loader(train.text.iloc[val], None, tokenizer, args) if val is not None else None
    model = new_model(args).to(device)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(param)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=args.learning_rate, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    updates_per_epoch = math.ceil(len(batches) / args.accumulation)
    # The same 3-epoch schedule horizon is used even when stopping at an earlier epoch.
    total_updates = updates_per_epoch * args.epochs
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
        skipped_updates = 0
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
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - started
                print(f"{directory.name} epoch {epoch}/{epochs}: {step+1}/{len(batches)} batches; "
                      f"{elapsed/(step+1):.2f}s/batch", flush=True)
        item = {"epoch": epoch, "training_loss": loss_sum / len(fit),
                "training_seconds": time.perf_counter() - started, "skipped_updates": skipped_updates}
        checkpoint = directory / f"epoch{epoch}"
        checkpoint.mkdir()
        tokenizer.save_pretrained(checkpoint)
        save_weights = keep_all or (validation is None and epoch == epochs)
        if validation is not None:
            scores = predict(model, validation, device)
            item.update(metrics(train.y.iloc[val], scores.argmax(axis=1)))
            np.save(checkpoint / "validation_logits.npy", scores)
            if item["macro_f1"] > best_f1:
                best_f1, best_epoch = item["macro_f1"], epoch
                save_weights = True
        else:
            best_epoch = epoch
        if save_weights:
            model.save_pretrained(checkpoint)
            if not keep_all:
                prune_unselected_weights(directory, best_epoch)
        history.append(item)
        write_json(directory / "history.json", history)
        print(json.dumps({k:v for k,v in item.items() if k not in ["classification_report", "confusion_matrix"]}), flush=True)
    result = {"selected_epoch": best_epoch, "class_weights": weights.tolist(), "history": history}
    write_json(directory / "complete.json", result)
    if not keep_all:
        prune_unselected_weights(directory, best_epoch)
    del optimizer, model, scheduler
    clear_device(device)
    return result


def save_predictions(path, ids, y, scores):
    frame = pd.DataFrame({"id": list(ids), "y": list(y), "y_pred": scores.argmax(axis=1)})
    frame.to_csv(path, index=False)
    np.save(path.with_suffix(".logits.npy"), scores)
    return frame


def probabilities(logits):
    values = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def evaluate_checkpoint(checkpoint, train, indices, tokenizer, args, destination):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to('cuda')
    values = predict(model, loader(train.text.iloc[indices], None, tokenizer, args), 'cuda')
    save_predictions(destination, train.id.iloc[indices], train.y.iloc[indices], values)
    result = metrics(train.y.iloc[indices], values.argmax(1))
    del model
    clear_device('cuda')
    return result


def worker(args, train, test, sample, run):
    require(torch.cuda.is_available(), 'Kaggle CUDA GPU required; Mac training is disabled')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    a, b, fit, holdout = make_splits(train)
    directory = run / f'seed{args.seed}'
    if args.worker_stage == 'select':
        directory.mkdir()
        train_phase(train, a, b, args.epochs, directory / 'inner', tokenizer, args, 'cuda', keep_all=True)
        return
    selection = json.loads((run / 'selection.json').read_text())
    epochs = selection['selected_epoch']
    if args.worker_stage == 'outer':
        train_phase(train, fit, None, epochs, directory / 'outer', tokenizer, args, 'cuda')
        result = evaluate_checkpoint(directory / f'outer/epoch{epochs}', train, holdout,
                                     tokenizer, args, directory / 'outer_predictions.csv')
        write_json(directory / 'outer_metrics.json', result)
    elif args.worker_stage == 'confirm':
        # Independent nested selection within fold 1; no fold 1 labels inform its fit/epoch choice.
        ca, cb, cfit, cval = make_splits(train, fold=1)
        confirm = run / 'confirmation'
        confirm.mkdir()
        np.savez(confirm / 'split_indices.npz', inner_train=ca, inner_validation=cb,
                 outer_train=cfit, outer_validation=cval)
        inner = train_phase(train, ca, cb, args.epochs, confirm / 'inner', tokenizer, args, 'cuda')
        ce = inner['selected_epoch']
        train_phase(train, cfit, None, ce, confirm / 'outer', tokenizer, args, 'cuda')
        result = evaluate_checkpoint(confirm / f'outer/epoch{ce}', train, cval,
                                     tokenizer, args, confirm / 'validation.csv')
        result.update(selected_epoch=ce, seed=args.seed, rows=len(cval),
                      limitation='Second grouped fold, single seed; not a second-fold ensemble estimate.')
        write_json(confirm / 'metrics.json', result)
    elif args.worker_stage == 'final':
        train_phase(train, np.arange(len(train)), None, epochs, directory / 'final', tokenizer, args, 'cuda')
        checkpoint = directory / f'final/epoch{epochs}'
        batches = loader(test.text, None, tokenizer, args)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to('cuda')
        scores = predict(model, batches, 'cuda')
        del model
        clear_device('cuda')
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to('cuda')
        verified = predict(model, batches, 'cuda')
        require(np.array_equal(scores.argmax(1), verified.argmax(1)), 'Reload changed labels')
        require(np.allclose(scores, verified, rtol=1e-4, atol=1e-4), 'Reload changed logits')
        np.save(directory / 'test_logits.npy', scores)
        write_json(directory / 'final_verified.json', {'reload_verified': True,
                   'checkpoint': str(checkpoint.relative_to(run)),
                   'weights_sha256': {p.name: digest(p) for p in checkpoint.glob('*.safetensors')}})
    else:
        raise ValueError(args.worker_stage)


def run_workers(tasks, args, run, gpu_count):
    """Separate CUDA processes use one visible GPU each; parent never trains a model."""
    pending = list(tasks)
    active = {}
    last_update = 0.
    try:
        while pending or active:
            for gpu in range(gpu_count):
                if gpu in active or not pending:
                    continue
                stage, seed = pending.pop(0)
                log_path = run / f'{stage}_seed{seed}.log'
                stream = log_path.open('x')
                env = os.environ.copy()
                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[gpu] if visible else str(gpu)
                env['TOKENIZERS_PARALLELISM'] = 'false'
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--root', str(args.root), '--data-dir', str(args.data_dir),
                       '--run-name', args.run_name, '--worker-stage', stage, '--seed', str(seed),
                       '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
                       '--accumulation', str(args.accumulation), '--max-length', str(args.max_length),
                       '--learning-rate', str(args.learning_rate)]
                proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (proc, stream, log_path)
                print(f'Started {stage} seed {seed} on GPU {gpu}', flush=True)
            for gpu, (proc, stream, path) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    stream.close()
                    del active[gpu]
                    if code:
                        print(path.read_text(errors='replace')[-12000:], flush=True)
                        raise RuntimeError(f'{path.name} failed with exit code {code}; preserved logs')
                    print(f'Finished {path.name}', flush=True)
            if time.monotonic() - last_update >= 30:
                for _, _, path in active.values():
                    lines = path.read_text(errors='replace').splitlines()
                    if lines:
                        print(f'{path.name}: {lines[-1]}', flush=True)
                last_update = time.monotonic()
            if active:
                time.sleep(5)
    finally:
        for proc, stream, _ in active.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            stream.close()


def combine_outer(run, train, holdout):
    combined = []
    individual = []
    for seed in SEEDS:
        path = run / f'seed{seed}'
        frame = pd.read_csv(path / 'outer_predictions.csv', dtype={'id': str})
        require(frame.id.tolist() == train.id.iloc[holdout].tolist(), 'Outer ID order changed')
        require(np.array_equal(frame.y, train.y.iloc[holdout]), 'Outer labels changed')
        logits = np.load(path / 'outer_predictions.logits.npy')
        require(np.array_equal(frame.y_pred, logits.argmax(1)), 'Outer logits mismatch')
        combined.append(probabilities(logits))
        individual.append({'seed': seed, **metrics(frame.y, frame.y_pred)})
    probs = np.mean(combined, axis=0)
    return probs, individual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root())
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--run-name', default='submission6')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=192)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--worker-stage', choices=['select', 'outer', 'confirm', 'final'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    require(args.run_name.replace('_', '').isalnum(), 'Invalid run name')
    require(min(args.epochs, args.batch_size, args.accumulation, args.max_length) > 0, 'Invalid settings')
    args.root = args.root.resolve()
    args.data_dir = (args.data_dir or find_data_dir(args.root)).resolve()
    args.pretrained = None
    train, test, sample, hashes = load_data(args.data_dir)
    run = args.root / 'models' / args.run_name
    if args.worker_stage:
        manifest = json.loads((run / 'manifest.json').read_text())
        require(manifest['source_sha256'] == digest(__file__), 'Worker source mismatch')
        require(manifest['input_sha256'] == hashes, 'Worker data mismatch')
        worker(args, train, test, sample, run)
        return
    require(torch.cuda.is_available(), 'Launch on Kaggle CUDA; local Mac training is disabled')
    gpu_count = min(2, torch.cuda.device_count())
    require(gpu_count == 2, 'Two GPUs required for the deadline; no silent serial fallback')
    report = args.root / 'reports' / f'{args.run_name}_metrics.json'
    output = args.root / 'submissions' / f'{args.run_name}_xlmr_large.csv'
    validation = args.root / 'data/processed' / f'{args.run_name}_validation.csv'
    for p in (run, report, output, validation):
        require(not p.exists(), f'Preserving {p}; choose a new run name')
    run.mkdir(parents=True)
    for p in (report.parent, output.parent, validation.parent):
        p.mkdir(parents=True, exist_ok=True)
    a, b, fit, holdout = make_splits(train)
    np.savez(run / 'split_indices.npz', inner_train=a, inner_validation=b, outer_train=fit, outer_validation=holdout)
    config = {k: getattr(args, k) for k in ('epochs', 'batch_size', 'accumulation', 'max_length', 'learning_rate')}
    manifest = {'model': MODEL, 'revision': REVISION, 'seeds': list(SEEDS), 'config': config,
                'input_sha256': hashes, 'source_sha256': digest(__file__), 'python': platform.python_version(),
                'versions': {p: importlib.metadata.version(p) for p in
                             ('torch', 'transformers', 'tokenizers', 'numpy', 'pandas', 'scikit-learn')},
                'devices': [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                'reference_submission5_macro_f1': REFERENCE_F1,
                'selection': 'Mean of two seed probabilities on inner validation; earliest best epoch',
                'confirmation': 'Fold 1 with its own nested epoch selection, seed 42; descriptive consistency check'}
    write_json(run / 'manifest.json', manifest)
    shutil.copy2(__file__, run / 'training_source.py')
    # Cache the initial weights once, outside /kaggle/working, before launching workers.
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION,
                      allow_patterns=['config.json', '*.safetensors', '*.safetensors.index.json',
                                      'tokenizer*.json', 'sentencepiece.bpe.model', 'special_tokens_map.json'])
    run_workers([('select', s) for s in SEEDS], args, run, gpu_count)
    trials = []
    for epoch in range(1, args.epochs + 1):
        values = [probabilities(np.load(run / f'seed{s}/inner/epoch{epoch}/validation_logits.npy')) for s in SEEDS]
        trials.append({'epoch': epoch, **metrics(train.y.iloc[b], np.mean(values, axis=0).argmax(1))})
    selected = max(trials, key=lambda t: (t['macro_f1'], -t['epoch']))
    epochs = selected['epoch']
    write_json(run / 'selection.json', {'selected_epoch': epochs, 'trials': trials})
    for seed in SEEDS:
        prune_unselected_weights(run / f'seed{seed}/inner', epochs)
    print(f'Selected epoch {epochs}; inner ensemble Macro F1 {selected["macro_f1"]:.6f}', flush=True)
    run_workers([('outer', s) for s in SEEDS], args, run, gpu_count)
    probs, individual = combine_outer(run, train, holdout)
    value = metrics(train.y.iloc[holdout], probs.argmax(1))
    pd.DataFrame({'id': train.id.iloc[holdout].to_numpy(), 'y': train.y.iloc[holdout].to_numpy(),
                  'y_pred': probs.argmax(1)}).to_csv(validation, index=False)
    np.save(run / 'outer_probabilities.npy', probs)
    result = {**value, 'manifest': manifest, 'selected_epoch': epochs, 'inner_trials': trials,
              'seed_outer_metrics': individual, 'gain_over_submission5': value['macro_f1'] - REFERENCE_F1,
              'status': 'evaluated', 'public_macro_f1': None,
              'limitation': 'Reused development folds and seed variation do not establish hidden-test stability.'}
    write_json(report, result)
    print(f'Outer ensemble Macro F1 {value["macro_f1"]:.6f}; reference {REFERENCE_F1:.6f}', flush=True)
    if value['macro_f1'] <= REFERENCE_F1:
        print('No improvement over submission 5. Evaluation preserved; final training skipped.', flush=True)
        return
    # Confirmation has its own selection. It runs alongside final refits to use both GPUs.
    run_workers([('confirm', 42), ('final', 42), ('final', 2026)], args, run, gpu_count)
    result['confirmation'] = json.loads((run / 'confirmation/metrics.json').read_text())
    scores = [probabilities(np.load(run / f'seed{s}/test_logits.npy')) for s in SEEDS]
    combined = np.mean(scores, axis=0)
    np.save(run / 'test_probabilities.npy', combined)
    lookup = pd.Series(combined.argmax(1), index=test.id)
    submission = sample[['id']].copy()
    submission['y_pred'] = submission.id.map(lookup)
    require(submission.id.is_unique and submission.id.equals(sample.id), 'ID/order mismatch')
    require(submission.y_pred.isin(LABELS).all(), 'Invalid labels')
    submission.to_csv(output, index=False, mode='x')
    require(pd.read_csv(output, dtype={'id': str}, keep_default_na=False).equals(submission), 'CSV roundtrip mismatch')
    result.update(status='ready_for_review', submission=str(output), submission_sha256=digest(output),
                  rows=len(submission), reload_verified=True,
                  review_note='Compare confirmation-fold and individual-seed results before manual upload.')
    write_json(report, result)
    print(f'Saved {output}; review confirmation metrics before manual submission.', flush=True)


if __name__ == '__main__':
    main()
