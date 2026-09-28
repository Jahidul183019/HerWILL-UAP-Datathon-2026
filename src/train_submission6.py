"""Submission 6: XLM-R-large blended with TF-IDF NB-SVM views.

The transformer follows submission 5's inner/outer/final protocol with the large
checkpoint. The three submission 4 NB-SVM views train in a parallel CPU process on
the same splits. Blend weight and class offsets are chosen on inner validation
only, then evaluated once on the outer holdout. A CSV is written only if the
blend beats submission 5's outer Macro F1.

Standalone script for Kaggle CUDA (or local MPS). Only competition CSVs are used.
Use a new run name for every experiment. Earlier submissions are never modified.
"""

from __future__ import annotations

import argparse
import itertools
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
    prepare_kaggle_environment()

import numpy as np
import pandas as pd
import torch
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


MODEL = "FacebookAI/xlm-roberta-large"
REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
LABELS = [0, 1, 2]
REFERENCE_F1 = 0.5590271875463054  # Submission 5 outer holdout (XLM-R-base).
BLEND_WEIGHTS = [round(0.05 * i, 2) for i in range(21)]  # Transformer share.
OFFSETS = [round(-0.3 + 0.05 * i, 2) for i in range(13)]  # Classes 1 and 2.
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
            "Competition inputs differ from the files used for submissions 1–5")
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


class NBSVM:
    """One-vs-rest LinearSVC with NB log-count ratios; same as src/nbsvm.py."""

    def __init__(self, C=0.1, alpha=1.0, random_state=42):
        self.C, self.alpha, self.random_state = C, alpha, random_state

    def fit(self, X, y):
        X, y = X.tocsr(), np.asarray(y)
        binary = X.copy()
        binary.data[:] = 1
        self.ratios_, self.estimators_ = [], []
        for label in LABELS:
            target = y == label
            positive = np.asarray(binary[target].sum(axis=0)).ravel() + self.alpha
            negative = np.asarray(binary[~target].sum(axis=0)).ravel() + self.alpha
            ratio = np.log((positive / positive.sum()) / (negative / negative.sum()))
            estimator = LinearSVC(C=self.C, class_weight="balanced", dual="auto",
                                  max_iter=10000, random_state=self.random_state)
            estimator.fit(X.multiply(ratio).tocsr(), target)
            self.ratios_.append(ratio)
            self.estimators_.append(estimator)
        return self

    def decision_function(self, X):
        return np.column_stack([e.decision_function(X.multiply(r).tocsr())
                                for r, e in zip(self.ratios_, self.estimators_)])


def nbsvm_views():
    """The three feature views selected for submission 4, with equal weights."""
    common = dict(min_df=2, sublinear_tf=True, lowercase=False, dtype=np.float32)
    word = lambda n, top: TfidfVectorizer(token_pattern=r"(?u)\S+", ngram_range=(1, n),
                                          max_features=top, **common)
    return {
        "word_char_wb": lambda: FeatureUnion([
            ("word", word(3, 160000)),
            ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6),
                                        max_features=240000, **common))]),
        "combined": lambda: FeatureUnion([
            ("word", word(2, 150000)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                     max_features=250000, **common))]),
        "char_wb": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6),
                                           max_features=300000, **common),
    }


def nbsvm_scores(fit_texts, fit_labels, score_texts):
    """Average of per-view NB-SVM scores, each scaled by its fitting-row std."""
    total = 0.0
    for name, make in nbsvm_views().items():
        started = time.perf_counter()
        vectorizer = make()
        X = vectorizer.fit_transform(fit_texts)
        model = NBSVM().fit(X, fit_labels)
        scale = max(float(model.decision_function(X).std()), 1e-8)
        total = total + model.decision_function(vectorizer.transform(score_texts)) / scale
        print(f"NB-SVM {name}: {X.shape[1]} features, {time.perf_counter()-started:.0f}s", flush=True)
        del vectorizer, X, model
        gc.collect()
    return total / 3


def run_nbsvm(train, test, splits, directory):
    """CPU worker: NB-SVM scores for inner validation, outer holdout and test."""
    texts, test_texts = train.text.map(normalize), test.text.map(normalize)
    a, b, fit, holdout = splits
    for name, rows, targets in (("inner", a, texts.iloc[b]), ("outer", fit, texts.iloc[holdout]),
                                ("final", np.arange(len(train)), test_texts)):
        path = directory / f"{name}_scores.npy"
        if not path.exists():
            np.save(path, nbsvm_scores(texts.iloc[rows], train.y.iloc[rows], targets))
    write_json(directory / "complete.json", {"views": list(nbsvm_views()), "C": 0.1, "alpha": 1.0})


def standardize(scores):
    """Label-free scaling so components are comparable on any scored set."""
    scores = np.asarray(scores, dtype=np.float64)
    return scores / max(float(scores.std()), 1e-8)


def blend(transformer_logits, nbsvm, weight, offsets):
    transformer = standardize(log_softmax(transformer_logits, axis=1))
    return weight * transformer + (1 - weight) * standardize(nbsvm) + np.array([0.0, *offsets])


def select_blend(y, transformer_logits, nbsvm):
    trials = []
    for weight in BLEND_WEIGHTS:
        for offsets in itertools.product(OFFSETS, OFFSETS):
            pred = blend(transformer_logits, nbsvm, weight, offsets).argmax(1)
            value = float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))
            trials.append({"weight": weight, "offsets": list(offsets), "macro_f1": value})
    # Ties prefer smaller offsets, then the more even blend.
    best = max(trials, key=lambda t: (round(t["macro_f1"], 10), -sum(map(abs, t["offsets"])),
                                      -abs(t["weight"] - 0.5)))
    return best, trials


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


def launch_nbsvm(args, root, data_dir, run):
    (run / "nbsvm").mkdir()
    log = (run / "nbsvm" / "worker.log").open("w")
    command = [sys.executable, "-u", __file__, "--stage", "nbsvm", "--run-name", args.run_name,
               "--root", str(root), "--data-dir", str(data_dir)]
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def wait_nbsvm(worker, run):
    print("Waiting for the NB-SVM worker", flush=True)
    code = worker.wait()
    require(code == 0 and (run / "nbsvm/complete.json").exists(),
            f"NB-SVM worker failed ({code}); see {run / 'nbsvm/worker.log'}")
    return {name: np.load(run / f"nbsvm/{name}_scores.npy") for name in ("inner", "outer", "final")}


def evaluate_transformer(train, rows, checkpoint, tokenizer, args, device):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    scores = predict(model, loader(train.text.iloc[rows], None, tokenizer, args), device)
    del model
    clear_device(device)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--run-name", default="submission6")
    parser.add_argument("--stage", choices=["all", "nbsvm"], default="all",
                        help="'nbsvm' is the internal CPU worker launched by 'all'")
    parser.add_argument("--pretrained", help="Optional local copy of the pinned initial checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    require(min(args.epochs, args.batch_size, args.accumulation, args.max_length) > 0, "Invalid training settings")
    root = args.root.resolve()
    data_dir = (args.data_dir or find_data_dir(root)).resolve()
    run = root / "models" / args.run_name
    report = root / "reports" / f"{args.run_name}_metrics.json"
    output = root / "submissions" / f"{args.run_name}_xlmr_large_nbsvm.csv"
    val_output = root / "data/processed" / f"{args.run_name}_validation.csv"
    train, test, sample, hashes = load_data(data_dir)
    splits = make_splits(train)
    a, b, fit, holdout = splits
    if args.stage == "nbsvm":
        require((run / "manifest.json").exists() and (run / "nbsvm").is_dir(), "Launch through --stage all")
        run_nbsvm(train, test, splits, run / "nbsvm")
        return
    for path in [run, report, output, val_output]:
        require(not path.exists(), f"Preserving {path}; choose a new --run-name")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    require(device != "cpu", "XLM-R-large training needs a GPU; use a Kaggle GPU runtime")
    torch.set_num_threads(2)  # Leave CPU cores for the NB-SVM worker.
    config = {k: getattr(args, k) for k in ["epochs", "batch_size", "accumulation", "max_length", "learning_rate", "seed"]}
    config.update(model=MODEL, revision=REVISION, loss="sqrt inverse frequency weighted cross entropy",
                  nbsvm_views=list(nbsvm_views()), blend_weights=BLEND_WEIGHTS, offsets=OFFSETS)
    if args.pretrained:
        initial_files = sorted(Path(args.pretrained).glob("*.safetensors"))
        require(bool(initial_files), "Local checkpoint must contain safetensors weights")
        config["initial_model_sha256"] = {p.name: digest(p) for p in initial_files}
    run.mkdir(parents=True)
    for directory in [report.parent, output.parent, val_output.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "input_sha256": hashes, "source_sha256": digest(__file__),
                "device": device, "precision": "CUDA mixed float16" if device == "cuda" else "float32",
                "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in
                    ["torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn"]},
                "rows": {"inner_train":len(a), "inner_validation":len(b), "outer_train":len(fit),
                         "outer_validation":len(holdout), "train":len(train), "test":len(test)},
                "reference_submission5_macro_f1":REFERENCE_F1,
                "preprocessing":"NFC and whitespace; original case for XLM-R; lowercased NB-SVM text and grouping"}
    write_json(run / "manifest.json", manifest)
    shutil.copy2(__file__, run / "training_source.py")
    np.savez(run / "split_indices.npz", inner_train=a, inner_validation=b, outer_train=fit, outer_validation=holdout)
    worker = launch_nbsvm(args, root, data_dir, run)
    try:
        kw = {} if args.pretrained else {"revision":REVISION}
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained or MODEL, **kw)
        print(f"Run {args.run_name}; device {device}; {manifest['rows']}", flush=True)
        selection = train_phase(train, a, b, args.epochs, run / "inner", tokenizer, args, device)
        epochs = selection["selected_epoch"]
        inner_f1 = selection["history"][epochs - 1]["macro_f1"]
        # A collapsed large model predicts one class (Macro F1 about 0.22); stop before spending more GPU.
        require(inner_f1 > 0.40, f"XLM-R-large inner Macro F1 {inner_f1:.4f}; training collapsed")
        inner_logits = np.load(run / f"inner/epoch{epochs}/validation_logits.npy")

        train_phase(train, fit, None, epochs, run / "outer", tokenizer, args, device)
        outer_logits = evaluate_transformer(train, holdout, run / f"outer/epoch{epochs}", tokenizer, args, device)
        nbsvm = wait_nbsvm(worker, run)
        y_inner, y_outer = train.y.iloc[b].to_numpy(), train.y.iloc[holdout].to_numpy()
        selected, trials = select_blend(y_inner, inner_logits, nbsvm["inner"])
        print(f"Inner blend: {selected}", flush=True)
        blended = blend(outer_logits, nbsvm["outer"], selected["weight"], selected["offsets"])
        save_predictions(val_output, train.id.iloc[holdout], y_outer, blended)
        np.save(run / "outer_transformer_logits.npy", outer_logits)
        result = metrics(y_outer, blended.argmax(1))
        components = {
            "xlmr_large": metrics(y_outer, outer_logits.argmax(1))["macro_f1"],
            "nbsvm_views": metrics(y_outer, nbsvm["outer"].argmax(1))["macro_f1"],
            "blend_without_offsets": metrics(y_outer, blend(outer_logits, nbsvm["outer"],
                                                             selected["weight"], [0, 0]).argmax(1))["macro_f1"],
        }
        result.update(manifest=manifest, selected_epoch=epochs, inner_history=selection["history"],
                      blend=selected, inner_nbsvm_macro_f1=metrics(y_inner, nbsvm["inner"].argmax(1))["macro_f1"],
                      outer_components=components, blend_trials=len(trials),
                      gain_over_submission5=result["macro_f1"] - REFERENCE_F1,
                      status="evaluated", limitation="Single reused outer holdout; not an independent final test.")
        write_json(report, result)
        print(f"Outer Macro F1: blend {result['macro_f1']:.6f} "
              f"(gain {result['gain_over_submission5']:+.6f}); components {components}", flush=True)
        if result["macro_f1"] <= REFERENCE_F1:
            print("No improvement over submission 5. Saved evaluation; no submission generated.", flush=True)
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
        final = blend(scores, nbsvm["final"], selected["weight"], selected["offsets"]).argmax(1)
        check = blend(verified, nbsvm["final"], selected["weight"], selected["offsets"]).argmax(1)
        require(np.array_equal(final, check), "Reload changed test predictions")
        lookup = pd.Series(final, index=test.id)
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
                      submission_rows=len(submission), final_model=str(checkpoint), reload_verified=True,
                      test_label_share=np.bincount(final, minlength=3).tolist())
        write_json(report, result)
        print(f"Saved {output}; public score unknown.", flush=True)
    finally:
        if worker.poll() is None:
            worker.kill()


if __name__ == "__main__":
    main()
