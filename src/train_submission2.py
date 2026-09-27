"""Select a second text model on an inner split, then evaluate the fixed holdout.

Run: python3 src/train_submission2.py
This script requires train_baseline.py and nbsvm.py from the same directory.
"""

import gc
import hashlib
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from nbsvm import NBSVM
from train_baseline import ROOT, LABELS, SEED, normalize, require


CANDIDATES = [
    {"kind": "svc", "C": 0.25},
    {"kind": "svc", "C": 1.0},
    {"kind": "nbsvm", "C": 0.25},
    {"kind": "nbsvm", "C": 1.0},
]


def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            token_pattern=r"(?u)\S+", ngram_range=(1, 2), min_df=2,
            max_features=150000, sublinear_tf=True, lowercase=False, dtype=np.float32)),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2,
            max_features=250000, sublinear_tf=True, lowercase=False, dtype=np.float32)),
    ])


def classifier(config):
    if config["kind"] == "nbsvm":
        return NBSVM(C=config["C"], random_state=SEED)
    return LinearSVC(C=config["C"], class_weight="balanced", dual="auto",
                     random_state=SEED, max_iter=10000)


def score(y, pred):
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))


def main():
    started = time.perf_counter()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    paths = {name: ROOT / f"data/raw/{name}.csv"
             for name in ("train", "test", "sample_submission")}
    train, test, sample = [pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
                           for path in paths.values()]
    require(train.columns.tolist() == ["id", "text", "y"], "Unexpected train schema")
    require(test.columns.tolist() == ["id", "text"], "Unexpected test schema")
    require(sample.columns.tolist() == ["id", "y_pred"], "Unexpected sample schema")
    for frame in (train, test, sample):
        require(frame.id.is_unique and frame.id.ne("").all(), "Invalid IDs")
    require(set(test.id) == set(sample.id), "Test and sample IDs differ")
    require(not set(train.id).intersection(test.id), "Train/test IDs overlap")
    require(set(train.y.unique()) == set(LABELS), "Invalid training labels")
    texts, test_texts = train.text.map(normalize), test.text.map(normalize)
    require(texts.ne("").all() and test_texts.ne("").all(), "Empty texts")
    groups = pd.factorize(texts, sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit, holdout = next(outer.split(texts, train.y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    inner_fit, inner_val = next(inner.split(texts.iloc[fit], train.y.iloc[fit], groups[fit]))
    inner_fit, inner_val = fit[inner_fit], fit[inner_val]
    require(not set(groups[fit]).intersection(groups[holdout]), "Outer group leakage")
    require(not set(groups[inner_fit]).intersection(groups[inner_val]), "Inner group leakage")
    print(f"Inner selection: {len(inner_fit)} fit / {len(inner_val)} validation; "
          f"outer holdout: {len(holdout)}", flush=True)
    vectorizer = features()
    X = vectorizer.fit_transform(texts.iloc[inner_fit])
    V = vectorizer.transform(texts.iloc[inner_val])
    results = []
    for config in CANDIDATES:
        estimator = classifier(config).fit(X, train.y.iloc[inner_fit])
        value = score(train.y.iloc[inner_val], estimator.predict(V))
        results.append({**config, "inner_macro_f1": value})
        print(f"Candidate {config}: {value:.6f}", flush=True)
    best = max(results, key=lambda row: row["inner_macro_f1"])
    del X, V, vectorizer, estimator
    gc.collect()
    print(f"Selected {best}; fitting outer training partition", flush=True)
    model = Pipeline([("features", features()), ("classifier", classifier(best))])
    model.fit(texts.iloc[fit], train.y.iloc[fit])
    pred = model.predict(texts.iloc[holdout])
    outer_score = score(train.y.iloc[holdout], pred)
    previous = pd.read_csv(ROOT / "data/processed/baseline_validation.csv", dtype={"id": str})
    require(previous.id.tolist() == train.id.iloc[holdout].tolist(), "Baseline holdout differs")
    require(np.array_equal(previous.y, train.y.iloc[holdout]), "Baseline labels differ")
    baseline_score = score(previous.y, previous.y_pred)
    print(f"Outer Macro F1: {outer_score:.6f}; baseline: {baseline_score:.6f}", flush=True)
    metrics = {
        "selection": "Four candidates on an inner grouped holdout; one selected before outer evaluation",
        "outer_split": {"n_splits": 5, "seed": SEED, "fold": 0, "fit_rows": len(fit),
                        "validation_rows": len(holdout)},
        "inner_split": {"n_splits": 4, "seed": 43, "fold": 0, "fit_rows": len(inner_fit),
                        "validation_rows": len(inner_val)},
        "candidates": results, "selected": best,
        "features": {"word_ngrams": [1, 2], "char_ngrams": [2, 5], "min_df": 2,
                     "max_word_features": 150000, "max_char_features": 250000},
        "macro_f1": outer_score, "baseline_macro_f1": baseline_score,
        "improvement": outer_score - baseline_score,
        "classification_report": classification_report(train.y.iloc[holdout], pred,
                                                        labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(train.y.iloc[holdout], pred, labels=LABELS).tolist(),
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    for folder in ("reports", "data/processed", "models", "submissions"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train.id.iloc[holdout].to_numpy(), "y": train.y.iloc[holdout].to_numpy(),
                  "y_pred": pred}).to_csv(ROOT / "data/processed/submission2_validation.csv", index=False)
    (ROOT / "reports/submission2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    require(outer_score > baseline_score, "Second model did not improve; no submission generated")
    del model
    gc.collect()
    print("Fitting selected model on all training rows", flush=True)
    final = Pipeline([("features", features()), ("classifier", classifier(best))])
    final.fit(texts, train.y)
    test_pred = pd.Series(final.predict(test_texts), index=test.id)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(test_pred)
    require(submission.y_pred.notna().all() and submission.y_pred.isin(LABELS).all(), "Invalid predictions")
    require(submission.id.equals(sample.id), "Submission order changed")
    stem = f"submission2_tfidf_{best['kind']}"
    submission.to_csv(ROOT / f"submissions/{stem}.csv", index=False)
    joblib.dump(final, ROOT / f"models/{stem}.joblib", compress=3)
    metrics.update({"submission": f"submissions/{stem}.csv", "test_rows": len(submission),
                    "elapsed_seconds": round(time.perf_counter() - started, 2)})
    (ROOT / "reports/submission2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {stem}.csv ({len(submission)} rows)", flush=True)


if __name__ == "__main__":
    main()
