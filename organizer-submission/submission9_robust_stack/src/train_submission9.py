"""Submission 9: robust stack of text models, ID-neighbour labels and neighbour text scores.

Changes from submission 8, aimed at private-leaderboard stability:

* The stacker trains on all rows with out-of-sample XLM-R logits (inner validation and
  outer holdout, 19,127 rows) instead of 9,564.
* Neighbour label features use every other training row as the pool, the same
  density the test rows see (submission 8 used a sparser pool during training).
* New features: the out-of-fold NB-SVM scores of neighbouring rows (train and test,
  so unlabeled neighbours also contribute), wider ID windows, and simple per-post
  script and length signals.
* The final prediction averages a logistic regression and a small, regularised
  gradient-boosting model. Class offsets are tuned by cross-validation, and the model
  is assessed by repeated nested 5-fold cross-validation rather than one holdout.

Inputs: submission 5 Kaggle outputs (XLM-R-base logits), submission 7 NB-SVM caches and
models/submission9/nbsvm_oof.npy from `nbsvm_oof_submission9.py`. CPU only.
Optional: `--large-dir` adds submission 6 XLM-R-large logits when that run is available.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_submission7 as s7

ROOT = s7.ROOT
LABEL_NEIGHBOURS = (1, 2, 5, 10, 25, 50, 100)
SCORE_NEIGHBOURS = (2, 10, 50)
OFFSETS = [round(-1 + 0.1 * i, 1) for i in range(21)]
REFERENCE_F1 = 0.6802978  # Submission 8, trained on inner rows and scored on the outer holdout.


def nearest(ids, pool_rows, query_rows, width):
    """Indices (into pool_rows) and ID distances of the `width` nearest pool rows, excluding self."""
    pool_ids = ids[pool_rows]
    order = np.argsort(pool_ids, kind="stable")
    sorted_ids = pool_ids[order]
    query = ids[query_rows]
    centre = np.searchsorted(sorted_ids, query)
    window = centre[:, None] + np.arange(-width - 1, width + 1)[None, :]
    valid = (window >= 0) & (window < len(sorted_ids))
    window = np.clip(window, 0, len(sorted_ids) - 1)
    distance = np.abs(sorted_ids[window] - query[:, None]).astype(float)
    distance[~valid | (sorted_ids[window] == query[:, None])] = np.inf
    pick = np.argsort(distance, axis=1, kind="stable")[:, :width]
    return order[np.take_along_axis(window, pick, 1)], np.take_along_axis(distance, pick, 1)


def build_features(ids, y, texts, scores, rows, label_pool, all_rows):
    """Per-row features; `scores` holds out-of-sample NB-SVM scores for every train and test row."""
    features = []
    index, distance = nearest(ids, label_pool, rows, max(LABEL_NEIGHBOURS))
    onehot = np.eye(3)[y[label_pool][index]]
    for k in LABEL_NEIGHBOURS:
        features.append(onehot[:, :k].mean(1))
    weight = 1 / (1 + distance[:, :10])
    features.append((onehot[:, :10] * weight[..., None]).sum(1) / weight.sum(1, keepdims=True))
    features.append(np.log1p(distance[:, :2]))
    index, _ = nearest(ids, all_rows, rows, max(SCORE_NEIGHBOURS))
    neighbour_scores = scores[all_rows][index]
    for k in SCORE_NEIGHBOURS:
        features.append(neighbour_scores[:, :k].mean(1))
    bangla = texts.str.contains("[ঀ-৿]").to_numpy(float)
    latin = texts.str.contains("[A-Za-z]").to_numpy(float)
    features.append(np.column_stack([bangla[rows], latin[rows],
                                     np.log1p(texts.str.len().to_numpy()[rows])]))
    features.append(bangla[all_rows][index[:, :10]].mean(1, keepdims=True))
    return np.hstack(features)


def text_features(parts):
    return np.hstack([s7.standardize(log_softmax(p, axis=1)) if kind == "logits" else s7.standardize(p)
                      for kind, p in parts])


def make_models():
    return [LogisticRegression(C=1.0, max_iter=5000),
            HistGradientBoostingClassifier(learning_rate=0.03, max_leaf_nodes=15, max_iter=300,
                                           l2_regularization=1.0, min_samples_leaf=50, random_state=0)]


def fit_predict(X, y, X_new):
    """Mean log-probabilities of both stackers plus offsets tuned by 3-fold CV on (X, y)."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cv = np.mean([np.log(cross_val_predict(m, X, y, cv=folds, method="predict_proba") + 1e-9)
                  for m in make_models()], axis=0)
    _, _, o1, o2 = max((s7.metrics(y, (cv + [0, o1, o2]).argmax(1))["macro_f1"], -abs(o1) - abs(o2), o1, o2)
                       for o1, o2 in itertools.product(OFFSETS, OFFSETS))
    offsets = np.array([0.0, o1, o2])
    new = np.mean([np.log(m.fit(X, y).predict_proba(X_new) + 1e-9) for m in make_models()], axis=0)
    return new + offsets, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission9")
    parser.add_argument("--xlmr-dir", type=Path, default=ROOT / "models/submission5_kaggle_results")
    parser.add_argument("--large-dir", type=Path, help="Downloaded submission 6 output (optional)")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require = s7.require
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    report = ROOT / "reports" / f"{args.run_name}_metrics.json"
    output = ROOT / "submissions" / f"{args.run_name}_robust_stack.csv"
    for path in [report, output]:
        require(not path.exists(), f"Preserving {path}; choose a new --run-name")
    paths = {name: ROOT / f"data/raw/{name}.csv" for name in s7.EXPECTED_INPUT_SHA256}
    hashes = {name: s7.digest(path) for name, path in paths.items()}
    require(hashes == s7.EXPECTED_INPUT_SHA256, "Competition inputs changed")
    train, test, sample = [pd.read_csv(p, dtype={"id": str}, keep_default_na=False) for p in paths.values()]
    a, b, fit, holdout = s7.make_splits(train)

    x = args.xlmr_dir
    epoch = json.loads((x / "reports/submission5_metrics.json").read_text())["selected_epoch"]
    parts = {"inner": [("logits", np.load(x / f"models/submission5/inner/epoch{epoch}/validation_logits.npy"))],
             "outer": [("logits", np.load(x / "data/processed/submission5_validation.logits.npy"))],
             "test": [("logits", np.load(x / "models/submission5/test_logits.npy"))]}
    if args.large_dir:
        g = args.large_dir
        large = json.loads((g / "reports/submission6_metrics.json").read_text())
        parts["inner"].append(("logits", np.load(g / f"models/submission6/inner/epoch{large['selected_epoch']}/validation_logits.npy")))
        parts["outer"].append(("logits", np.load(g / "models/submission6/outer_transformer_logits.npy")))
        parts["test"].append(("logits", np.load(g / "models/submission6/test_logits.npy")))
    cache = ROOT / "models/submission7"
    for name, file in (("inner", "inner"), ("outer", "outer"), ("test", "final")):
        parts[name].append(("scores", np.load(cache / f"nbsvm_{file}.npy")))

    ids = np.concatenate([train.id.astype(int).to_numpy(), test.id.astype(int).to_numpy()])
    require(len(np.unique(ids)) == len(ids), "Duplicate IDs across train and test")
    y = np.concatenate([train.y.to_numpy(), np.full(len(test), -1)])
    texts = pd.concat([train.text, test.text], ignore_index=True).map(s7.normalize)
    oof = np.load(ROOT / "models/submission9/nbsvm_oof.npy")
    scores = np.vstack([s7.standardize(oof), s7.standardize(np.load(cache / "nbsvm_final.npy"))])
    train_rows, all_rows = np.arange(len(train)), np.arange(len(ids))
    test_rows = np.arange(len(train), len(ids))

    def rows_features(name, rows):
        return np.hstack([text_features(parts[name]),
                          build_features(ids, y, texts, scores, rows, train_rows, all_rows)])

    F_inner, F_outer, F_test = rows_features("inner", b), rows_features("outer", holdout), rows_features("test", test_rows)
    y_inner, y_outer = y[b], y[holdout]

    # Check 1: same protocol as submission 8 (fit on inner rows, score the outer holdout once).
    outer_scores, _ = fit_predict(F_inner, y_inner, F_outer)
    outer_f1 = s7.metrics(y_outer, outer_scores.argmax(1))["macro_f1"]
    print(f"Inner -> outer Macro F1 {outer_f1:.4f} (submission 8: {REFERENCE_F1:.4f})", flush=True)

    # Check 2: repeated nested 5-fold CV over all 19,127 stacking rows.
    F, Y = np.vstack([F_inner, F_outer]), np.concatenate([y_inner, y_outer])
    cv_scores = []
    for repeat in range(args.repeats):
        oof_pred = np.zeros(len(Y), dtype=int)
        for fit_rows, val_rows in StratifiedKFold(5, shuffle=True, random_state=repeat).split(F, Y):
            pred, _ = fit_predict(F[fit_rows], Y[fit_rows], F[val_rows])
            oof_pred[val_rows] = pred.argmax(1)
        cv_scores.append(s7.metrics(Y, oof_pred)["macro_f1"])
        print(f"Repeat {repeat}: nested CV Macro F1 {cv_scores[-1]:.4f}", flush=True)

    final_scores, offsets = fit_predict(F, Y, F_test)
    final = final_scores.argmax(1)
    result = s7.metrics(y_outer, outer_scores.argmax(1))
    result.update(model="LR + HGB stack: XLM-R logits, NB-SVM, ID-neighbour labels and neighbour NB-SVM scores",
                  inner_to_outer_macro_f1=outer_f1, nested_cv_macro_f1=cv_scores,
                  nested_cv_mean=float(np.mean(cv_scores)), nested_cv_std=float(np.std(cv_scores)),
                  final_offsets=offsets.tolist(), uses_xlmr_large=bool(args.large_dir),
                  stacking_rows=len(Y), input_sha256=hashes, source_sha256=s7.digest(__file__),
                  limitation="Relies on ID order carrying source information.")
    report.write_text(json.dumps(result, indent=2) + "\n")
    require(outer_f1 > REFERENCE_F1, f"Outer Macro F1 {outer_f1:.4f} does not beat submission 8")
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(pd.Series(final, index=test.id))
    require(submission.id.equals(sample.id) and submission.y_pred.isin(s7.LABELS).all(), "Invalid submission")
    submission.to_csv(output, index=False, mode="x")
    result.update(status="ready_for_manual_submission", submission=str(output.relative_to(ROOT)),
                  submission_sha256=s7.digest(output), test_label_share=np.bincount(final, minlength=3).tolist())
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output.name}; nested CV {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
