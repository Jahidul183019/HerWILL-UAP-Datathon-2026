"""Submission 8: stack text-model scores with ID-neighbour label features.

The competition IDs are not random: consecutive IDs usually come from the same
source (98% share the same script) and have correlated labels, and test IDs are
interleaved with training IDs. For every row this script summarises the labels of
the nearest *labelled* rows by ID (excluding itself) and stacks those features with
submission 7's inputs (submission 5 XLM-R-base logits and the NB-SVM views) in a
multinomial logistic regression.

Leakage control: the stacker trains on inner-validation rows using neighbours from
the outer-fit rows only (never the outer holdout), is evaluated once on the outer
holdout with the same neighbour pool, and at test time uses all training rows as
neighbours. Class offsets are chosen by 5-fold cross-validation on the inner rows.

Requires the outputs of `train_submission7.py` (cached NB-SVM scores) and the
downloaded submission 5 Kaggle run. CPU only; about one minute once those exist.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import train_submission7 as s7

ROOT = s7.ROOT
REFERENCE_F1 = 0.5798244649952974  # Submission 7 outer holdout.
NEIGHBOURS = (1, 2, 5, 10, 25)
OFFSETS = [round(-1 + 0.1 * i, 1) for i in range(21)]


def neighbour_features(ids, y, label_rows, query_rows):
    """Label shares of the k nearest labelled rows by ID, excluding the row itself."""
    labelled = ids[label_rows]
    order = np.argsort(labelled)
    labelled, labels = labelled[order], y[label_rows][order]
    query = ids[query_rows]
    width = max(NEIGHBOURS)
    window = np.clip(np.searchsorted(labelled, query)[:, None] + np.arange(-width, width)[None, :],
                     0, len(labelled) - 1)
    distance = np.abs(labelled[window] - query[:, None]).astype(float)
    distance[labelled[window] == query[:, None]] = 1e9  # Never use a row's own label.
    # Clipping at the edges repeats indices; drop repeats by pushing them to the end.
    sorted_window = np.sort(window, axis=1)
    repeat = np.zeros_like(window, dtype=bool)
    repeat[:, 1:] = sorted_window[:, 1:] == sorted_window[:, :-1]
    window, distance = sorted_window, np.take_along_axis(distance, np.argsort(window, axis=1), 1)
    distance[repeat] = 1e9
    nearest = np.argsort(distance, axis=1, kind="stable")
    window = np.take_along_axis(window, nearest, 1)
    distance = np.take_along_axis(distance, nearest, 1)
    onehot = np.eye(3)[labels[window]]
    features = [onehot[:, :k].mean(1) for k in NEIGHBOURS]
    weight = 1 / (1 + distance[:, :10])
    features.append((onehot[:, :10] * weight[..., None]).sum(1) / weight.sum(1, keepdims=True))
    features.append(np.log1p(distance[:, :1]))
    return np.hstack(features)


def text_features(logits, nbsvm):
    return np.hstack([s7.standardize(log_softmax(logits, axis=1)), s7.standardize(nbsvm)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission8")
    parser.add_argument("--submission7-run", default="submission7")
    parser.add_argument("--xlmr-dir", type=Path, default=ROOT / "models/submission5_kaggle_results")
    args = parser.parse_args()
    require = s7.require
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    report = ROOT / "reports" / f"{args.run_name}_metrics.json"
    output = ROOT / "submissions" / f"{args.run_name}_id_neighbour_stack.csv"
    val_output = ROOT / "data/processed" / f"{args.run_name}_validation.csv"
    for path in [report, output, val_output]:
        require(not path.exists(), f"Preserving {path}; choose a new --run-name")
    paths = {name: ROOT / f"data/raw/{name}.csv" for name in s7.EXPECTED_INPUT_SHA256}
    hashes = {name: s7.digest(path) for name, path in paths.items()}
    require(hashes == s7.EXPECTED_INPUT_SHA256, "Competition inputs changed")
    train, test, sample = [pd.read_csv(p, dtype={"id": str}, keep_default_na=False) for p in paths.values()]
    a, b, fit, holdout = s7.make_splits(train)
    require(pd.Series(train.id.tolist() + test.id.tolist()).str.fullmatch(r"\d+").all(), "Non-numeric IDs")

    x, cache = args.xlmr_dir, ROOT / "models" / args.submission7_run
    epoch = json.loads((x / "reports/submission5_metrics.json").read_text())["selected_epoch"]
    inner_logits = np.load(x / f"models/submission5/inner/epoch{epoch}/validation_logits.npy")
    outer_logits = np.load(x / "data/processed/submission5_validation.logits.npy")
    test_logits = np.load(x / "models/submission5/test_logits.npy")
    nb = {name: np.load(cache / f"nbsvm_{name}.npy") for name in ("inner", "outer", "final")}
    require(nb["inner"].shape == inner_logits.shape and nb["outer"].shape == outer_logits.shape
            and nb["final"].shape == test_logits.shape, "Cached score shapes differ")

    ids = np.concatenate([train.id.astype(int).to_numpy(), test.id.astype(int).to_numpy()])
    y = np.concatenate([train.y.to_numpy(), np.full(len(test), -1)])
    test_rows = np.arange(len(train), len(ids))
    train_features = np.hstack([text_features(inner_logits, nb["inner"]), neighbour_features(ids, y, fit, b)])
    outer_features = np.hstack([text_features(outer_logits, nb["outer"]),
                                neighbour_features(ids, y, fit, holdout)])
    test_features = np.hstack([text_features(test_logits, nb["final"]),
                               neighbour_features(ids, y, np.arange(len(train)), test_rows)])
    y_inner, y_outer = train.y.iloc[b].to_numpy(), train.y.iloc[holdout].to_numpy()

    stacker = LogisticRegression(C=1.0, max_iter=5000)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_predict(stacker, train_features, y_inner, cv=folds, method="predict_log_proba")
    trials = [(s7.metrics(y_inner, (cv_scores + [0, o1, o2]).argmax(1))["macro_f1"], -abs(o1) - abs(o2), o1, o2)
              for o1, o2 in itertools.product(OFFSETS, OFFSETS)]
    cv_f1, _, o1, o2 = max(trials)
    offsets = np.array([0.0, o1, o2])
    stacker.fit(train_features, y_inner)
    outer_pred = (stacker.predict_log_proba(outer_features) + offsets).argmax(1)
    pd.DataFrame({"id": train.id.iloc[holdout].to_numpy(), "y": y_outer, "y_pred": outer_pred}).to_csv(
        val_output, index=False)
    neighbour_only = neighbour_features(ids, y, fit, holdout)[:, 15:18].argmax(1)  # Distance-weighted k=10.
    result = s7.metrics(y_outer, outer_pred)
    result.update(model="logistic regression over XLM-R-base, NB-SVM and ID-neighbour label features",
                  offsets=offsets.tolist(), inner_cv_macro_f1=cv_f1,
                  outer_components={"submission7_blend": REFERENCE_F1,
                                    "id_neighbours_only": s7.metrics(y_outer, neighbour_only)["macro_f1"]},
                  gain_over_submission7=result["macro_f1"] - REFERENCE_F1, input_sha256=hashes,
                  source_sha256=s7.digest(__file__), neighbours=list(NEIGHBOURS), status="evaluated",
                  limitation="Relies on ID order carrying source information; single reused outer holdout.")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Outer Macro F1 {result['macro_f1']:.6f} (gain {result['gain_over_submission7']:+.6f}); "
          f"inner CV {cv_f1:.4f}; offsets {offsets.tolist()}", flush=True)
    if result["macro_f1"] <= REFERENCE_F1:
        print("No improvement over submission 7; no submission generated.", flush=True)
        return
    final = (stacker.predict_log_proba(test_features) + offsets).argmax(1)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(pd.Series(final, index=test.id))
    require(submission.id.equals(sample.id) and submission.y_pred.isin(s7.LABELS).all(), "Invalid submission")
    submission.to_csv(output, index=False, mode="x")
    result.update(status="ready_for_manual_submission", submission=str(output.relative_to(ROOT)),
                  submission_sha256=s7.digest(output), submission_rows=len(submission),
                  test_label_share=np.bincount(final, minlength=3).tolist())
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output.name}", flush=True)


if __name__ == "__main__":
    main()
