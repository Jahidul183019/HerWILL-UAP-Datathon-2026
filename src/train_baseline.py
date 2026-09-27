"""Reproducible TF-IDF + LinearSVC baseline; run from any working directory.

Only competition training text is used to fit features and the classifier.
The first of five stratified group folds is a fixed validation holdout.
Normalized duplicate texts stay in the same fold, including conflicting labels.
After evaluation, a fresh pipeline fits all training rows and predicts test IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import unicodedata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
LABELS = [0, 1, 2]
SEED = 42


def normalize(text: str) -> str:
    # Keep punctuation, emoji, and script distinctions; normalize Unicode/spacing.
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def make_model() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", token_pattern=r"(?u)\S+", ngram_range=(1, 2),
                min_df=2, max_features=80000, sublinear_tf=True,
                lowercase=False, dtype=np.float32,
            )),
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=150000, sublinear_tf=True,
                lowercase=False, dtype=np.float32,
            )),
        ])),
        ("classifier", LinearSVC(
            C=1.0, class_weight="balanced", random_state=SEED,
            dual="auto", max_iter=5000,
        )),
    ])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    started = time.perf_counter()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    paths = {name: args.data_dir / f"{name}.csv"
             for name in ("train", "test", "sample_submission")}
    frames = {name: pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
              for name, path in paths.items()}
    train, test, sample = (frames[n] for n in paths)
    for name, columns in (("train", ["id", "text", "y"]),
                          ("test", ["id", "text"]),
                          ("sample_submission", ["id", "y_pred"])):
        frame = frames[name]
        require(frame.columns.tolist() == columns, f"Unexpected columns in {name}")
        require(frame.id.is_unique and frame.id.ne("").all(), f"Invalid IDs in {name}")
    require(set(train.y.unique()) == set(LABELS), "Training labels must be 0, 1, 2")
    require(set(test.id) == set(sample.id), "Test and sample IDs differ")
    require(not set(train.id).intersection(test.id), "Train/test IDs overlap")
    texts = train.text.map(normalize)
    test_texts = test.text.map(normalize)
    require(texts.ne("").all() and test_texts.ne("").all(), "Empty normalized texts")
    groups = pd.factorize(texts, sort=True)[0]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(texts, train.y, groups))
    require(not set(groups[fit_idx]).intersection(groups[val_idx]), "Group leakage")

    print(f"Fitting baseline: {len(fit_idx):,} train / {len(val_idx):,} validation", flush=True)
    model = make_model()
    model.fit(texts.iloc[fit_idx], train.y.iloc[fit_idx])
    predictions = model.predict(texts.iloc[val_idx])
    score = f1_score(train.y.iloc[val_idx], predictions, labels=LABELS, average="macro")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(fit_idx), 1)), train.y.iloc[fit_idx])
    dummy_score = f1_score(train.y.iloc[val_idx], dummy.predict(np.zeros((len(val_idx), 1))),
                          labels=LABELS, average="macro", zero_division=0)
    report = classification_report(train.y.iloc[val_idx], predictions,
                                   labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(train.y.iloc[val_idx], predictions, labels=LABELS)
    print(f"Validation Macro F1: {score:.6f}; majority baseline: {dummy_score:.6f}", flush=True)
    for directory in ("reports", "models", "submissions", "data/processed"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train.id.iloc[val_idx].to_numpy(),
                  "y": train.y.iloc[val_idx].to_numpy(), "y_pred": predictions}).to_csv(
        ROOT / "data/processed/baseline_validation.csv", index=False)

    print("Refitting a fresh model on all training rows...", flush=True)
    final_model = make_model()
    final_model.fit(texts, train.y)
    test_predictions = pd.Series(final_model.predict(test_texts), index=test.id)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(test_predictions)
    require(submission.y_pred.notna().all(), "Missing predictions")
    require(submission.y_pred.isin(LABELS).all(), "Invalid predictions")
    require(submission.id.equals(sample.id), "Submission order mismatch")
    submission.to_csv(ROOT / "submissions/submission1_tfidf_svc.csv", index=False)
    # Normalization is applied before pipeline inference; keep it in this script.
    joblib.dump(final_model, ROOT / "models/baseline_tfidf_svc.joblib", compress=3)
    counts = lambda values: {str(k): int(v) for k, v in values.value_counts().sort_index().items()}
    metrics = {
        "model": "word + character TF-IDF / balanced LinearSVC", "seed": SEED,
        "validation": "First fold of 5-fold StratifiedGroupKFold; normalized text groups",
        "hyperparameters": {"C": 1.0, "class_weight": "balanced", "max_iter": 5000,
                            "word_ngrams": [1, 2], "char_ngrams": [3, 5],
                            "word_max_features": 80000, "char_max_features": 150000,
                            "min_df": 2, "sublinear_tf": True},
        "rows": {"train_total": len(train), "fit": len(fit_idx),
                 "validation": len(val_idx), "test": len(test)},
        "class_counts": {"all": counts(train.y), "fit": counts(train.y.iloc[fit_idx]),
                         "validation": counts(train.y.iloc[val_idx]),
                         "test_predictions": counts(submission.y_pred)},
        "normalized_duplicate_rows": int(texts.duplicated().sum()),
        "conflicting_label_groups": int(train.groupby(texts).y.nunique().gt(1).sum()),
        "test_texts_also_in_train": int(test_texts.isin(set(texts)).sum()),
        "macro_f1": score, "majority_macro_f1": dummy_score,
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (ROOT / "reports/baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model, aggregate metrics, and {len(submission):,} test predictions.", flush=True)
    print(f"Elapsed: {metrics['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()
