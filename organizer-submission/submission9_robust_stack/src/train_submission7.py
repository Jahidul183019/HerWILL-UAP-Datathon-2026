"""Submission 7: submission 5's XLM-R-base logits blended with TF-IDF NB-SVM views.

CPU only. Reuses the XLM-R-base validation and test logits saved by the submission 5
Kaggle run (download with `kaggle kernels output jahidul183019/submission-5-xlm-r`).
NB-SVM views are fitted on the same splits. Blend weight and class offsets are chosen
on inner validation only, evaluated once on the outer holdout, and a CSV is written
only if the blend beats submission 5's outer Macro F1.
"""

import argparse
import gc
import hashlib
import itertools
import json
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
LABELS = [0, 1, 2]
REFERENCE_F1 = 0.5590271875463054  # Submission 5 outer holdout (XLM-R-base alone).
SUBMISSION5_SHA256 = "29b866e0315a8cab515bc9d1c8e5397a2418ed3d033cd091f41862e548c19419"
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


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


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


def metrics(y, predictions):
    return {"macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0)),
            "classification_report": classification_report(y, predictions, labels=LABELS,
                                                             output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist()}


def run_nbsvm_phase(texts, labels, rows, targets, path):
    if path.exists():
        return np.load(path)
    scores = nbsvm_scores(texts.iloc[rows], labels.iloc[rows], targets)
    np.save(path, scores)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission7")
    parser.add_argument("--xlmr-dir", type=Path, default=ROOT / "models/submission5_kaggle_results",
                        help="Downloaded output of the submission 5 Kaggle run")
    parser.add_argument("--allow-new-xlmr", action="store_true",
                        help="Accept XLM-R outputs from a fresh submission 5 run instead of the recorded one")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    run = ROOT / "models" / args.run_name
    report = ROOT / "reports" / f"{args.run_name}_metrics.json"
    output = ROOT / "submissions" / f"{args.run_name}_xlmr_base_nbsvm.csv"
    val_output = ROOT / "data/processed" / f"{args.run_name}_validation.csv"
    for path in [report, output, val_output]:
        require(not path.exists(), f"Preserving {path}; choose a new --run-name")
    started = time.perf_counter()
    paths = {name: ROOT / f"data/raw/{name}.csv" for name in EXPECTED_INPUT_SHA256}
    hashes = {name: digest(path) for name, path in paths.items()}
    require(hashes == EXPECTED_INPUT_SHA256, "Competition inputs changed")
    train, test, sample = [pd.read_csv(p, dtype={"id": str}, keep_default_na=False) for p in paths.values()]
    a, b, fit, holdout = make_splits(train)

    # Submission 5 artifacts: selected inner epoch, outer holdout and final test logits.
    x = args.xlmr_dir
    s5 = json.loads((x / "reports/submission5_metrics.json").read_text())
    epoch = s5["selected_epoch"]
    inner_logits = np.load(x / f"models/submission5/inner/epoch{epoch}/validation_logits.npy")
    outer_logits = np.load(x / "data/processed/submission5_validation.logits.npy")
    test_logits = np.load(x / "models/submission5/test_logits.npy")
    outer_ids = pd.read_csv(x / "data/processed/submission5_validation.csv", dtype={"id": str}).id
    require(outer_ids.tolist() == train.id.iloc[holdout].tolist(), "Submission 5 holdout differs")
    require(inner_logits.shape == (len(b), 3) and test_logits.shape == (len(test), 3), "Logit shapes differ")
    require(args.allow_new_xlmr or
            abs(metrics(train.y.iloc[holdout], outer_logits.argmax(1))["macro_f1"] - REFERENCE_F1) < 1e-9,
            "Submission 5 outer logits do not reproduce its score")
    s5_csv = x / "submissions/submission5_xlmr.csv"
    require(args.allow_new_xlmr or digest(s5_csv) == SUBMISSION5_SHA256, "Submission 5 CSV differs")
    s5_pred = pd.read_csv(s5_csv, dtype={"id": str}).set_index("id").y_pred
    require((s5_pred.loc[test.id].to_numpy() == test_logits.argmax(1)).all(), "Test logits misaligned")

    run.mkdir(parents=True, exist_ok=True)
    for directory in [report.parent, output.parent, val_output.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    texts, test_texts = train.text.map(normalize), test.text.map(normalize)
    nb_inner = run_nbsvm_phase(texts, train.y, a, texts.iloc[b], run / "nbsvm_inner.npy")
    nb_outer = run_nbsvm_phase(texts, train.y, fit, texts.iloc[holdout], run / "nbsvm_outer.npy")

    y_inner, y_outer = train.y.iloc[b].to_numpy(), train.y.iloc[holdout].to_numpy()
    selected, trials = select_blend(y_inner, inner_logits, nb_inner)
    print(f"Inner blend: {selected}", flush=True)
    blended = blend(outer_logits, nb_outer, selected["weight"], selected["offsets"])
    pd.DataFrame({"id": train.id.iloc[holdout].to_numpy(), "y": y_outer,
                  "y_pred": blended.argmax(1)}).to_csv(val_output, index=False)
    result = metrics(y_outer, blended.argmax(1))
    components = {
        "xlmr_base": metrics(y_outer, outer_logits.argmax(1))["macro_f1"],
        "nbsvm_views": metrics(y_outer, nb_outer.argmax(1))["macro_f1"],
        "blend_without_offsets": metrics(y_outer, blend(outer_logits, nb_outer, selected["weight"],
                                                         [0, 0]).argmax(1))["macro_f1"],
    }
    result.update(model="submission 5 XLM-R-base logits + three TF-IDF NB-SVM views",
                  blend=selected, blend_trials=len(trials), outer_components=components,
                  gain_over_submission5=result["macro_f1"] - REFERENCE_F1, input_sha256=hashes,
                  source_sha256=digest(__file__), status="evaluated",
                  limitation="Single reused outer holdout; not an independent final test.")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Outer Macro F1: blend {result['macro_f1']:.6f} (gain {result['gain_over_submission5']:+.6f}); "
          f"components {components}", flush=True)
    if result["macro_f1"] <= REFERENCE_F1:
        print("No improvement over submission 5; no submission generated.", flush=True)
        return

    nb_test = run_nbsvm_phase(texts, train.y, np.arange(len(train)), test_texts, run / "nbsvm_final.npy")
    final = blend(test_logits, nb_test, selected["weight"], selected["offsets"]).argmax(1)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(pd.Series(final, index=test.id))
    require(submission.id.equals(sample.id) and submission.y_pred.isin(LABELS).all(), "Invalid submission")
    submission.to_csv(output, index=False, mode="x")
    result.update(status="ready_for_manual_submission", submission=str(output.relative_to(ROOT)),
                  submission_sha256=digest(output), submission_rows=len(submission),
                  test_label_share=np.bincount(final, minlength=3).tolist(),
                  changed_vs_submission5=int((submission.y_pred.to_numpy() !=
                                              s5_pred.loc[sample.id].to_numpy()).sum()),
                  elapsed_seconds=round(time.perf_counter() - started, 1))
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output.name}; {result['changed_vs_submission5']} labels differ from submission 5", flush=True)


if __name__ == "__main__":
    main()
