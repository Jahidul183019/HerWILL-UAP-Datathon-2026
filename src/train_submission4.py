"""Train a CPU ensemble with complementary TF-IDF views for submission 4."""

import argparse
import gc
import hashlib
import itertools
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
from sklearn.pipeline import FeatureUnion

from nbsvm import NBSVM
from train_baseline import ROOT, LABELS, SEED, normalize, require
from train_submission2 import features as submission2_features


CANDIDATES = [
    {"name": "combined", "view": "combined", "C": 0.10},
    {"name": "word", "view": "word", "C": 0.10},
    {"name": "char_wb", "view": "char_wb", "C": 0.10},
    {"name": "char_long", "view": "char_long", "C": 0.10},
    {"name": "word_char_wb", "view": "word_char_wb", "C": 0.10},
]


def make_features(view):
    common = dict(min_df=2, sublinear_tf=True, lowercase=False, dtype=np.float32)
    if view == "combined":
        return submission2_features()
    if view == "word":
        return TfidfVectorizer(token_pattern=r"(?u)\S+", ngram_range=(1, 3),
                               max_features=220000, **common)
    if view == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6),
                               max_features=300000, **common)
    if view == "char_long":
        return TfidfVectorizer(analyzer="char", ngram_range=(3, 6),
                               max_features=300000, **common)
    if view == "word_char_wb":
        return FeatureUnion([
            ("word", TfidfVectorizer(token_pattern=r"(?u)\S+", ngram_range=(1, 3),
                                     max_features=160000, **common)),
            ("char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6),
                                        max_features=240000, **common)),
        ])
    raise ValueError(view)


def macro_f1(y, pred):
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))


def fit_candidate(config, texts, labels):
    vectorizer = make_features(config["view"])
    X = vectorizer.fit_transform(texts)
    model = NBSVM(C=config["C"], alpha=1.0, random_state=SEED).fit(X, labels)
    scale = max(float(model.decision_function(X).std()), 1e-8)
    return {"config": config, "vectorizer": vectorizer, "model": model, "scale": scale}


def decision(member, texts):
    X = member["vectorizer"].transform(texts)
    return member["model"].decision_function(X) / member["scale"]


def ensemble_scores(scores, indices, weights, offsets):
    result = sum(float(weight) * scores[index] for index, weight in zip(indices, weights))
    return result + np.asarray(offsets)


def fit_members(configs, texts, labels):
    members = []
    for config in configs:
        member = fit_candidate(config, texts, labels)
        members.append(member)
        print(f"Fitted {config['name']}: {member['vectorizer'].transform(texts.iloc[:1]).shape[1]} features",
              flush=True)
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission4",
                        help="Unique output prefix; use a new name for each run")
    parser.add_argument("--reference-validation", default="data/processed/submission3_validation.csv",
                        help="Submission 3 validation CSV, relative to repository root or absolute")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Use letters, digits and underscores for run name")
    name = args.run_name
    inner_path = ROOT / f"models/{name}_inner_candidates.joblib"
    validation_model_path = ROOT / f"models/{name}_validation_model.joblib"
    validation_path = ROOT / f"data/processed/{name}_validation.csv"
    report_path = ROOT / f"reports/{name}_metrics.json"
    output = ROOT / f"submissions/{name}_tfidf_nbsvm_views.csv"
    model_path = ROOT / f"models/{name}_tfidf_nbsvm_views.joblib"
    output_paths = (inner_path, validation_model_path, validation_path, report_path, output, model_path)
    for path in output_paths:
        require(not path.exists(), f"Preserving existing {path}; choose a new --run-name")
    started = time.perf_counter()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    paths = {name: ROOT / f"data/raw/{name}.csv"
             for name in ("train", "test", "sample_submission")}
    train, test, sample = [pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
                           for path in paths.values()]
    for frame, columns in ((train, ["id", "text", "y"]), (test, ["id", "text"]),
                           (sample, ["id", "y_pred"])):
        require(frame.columns.tolist() == columns, "Unexpected schema")
        require(frame.id.is_unique and frame.id.ne("").all(), "Invalid IDs")
    require(set(train.y.unique()) == set(LABELS), "Invalid labels")
    require(set(test.id) == set(sample.id), "Test/sample IDs differ")
    require(not set(train.id).intersection(test.id), "Train/test IDs overlap")
    texts, test_texts = train.text.map(normalize), test.text.map(normalize)
    require(texts.ne("").all() and test_texts.ne("").all(), "Empty normalized texts")
    groups = pd.factorize(texts.str.lower(), sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit, holdout = next(outer.split(texts, train.y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    inner_fit, inner_validation = next(inner.split(texts.iloc[fit], train.y.iloc[fit], groups[fit]))
    inner_fit, inner_validation = fit[inner_fit], fit[inner_validation]
    require(not set(groups[inner_fit]).intersection(groups[inner_validation]), "Inner leakage")
    require(not set(groups[fit]).intersection(groups[holdout]), "Outer leakage")
    previous = pd.read_csv(ROOT / args.reference_validation, dtype={"id": str})
    require(previous.id.tolist() == train.id.iloc[holdout].tolist(), "Submission 3 holdout differs")
    require(np.array_equal(previous.y, train.y.iloc[holdout]), "Submission 3 labels differ")
    previous_score = macro_f1(previous.y, previous.y_pred)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Inner: {len(inner_fit)} fit / {len(inner_validation)} validation", flush=True)
    inner_members = fit_members(CANDIDATES, texts.iloc[inner_fit], train.y.iloc[inner_fit])
    inner_scores = [decision(member, texts.iloc[inner_validation]) for member in inner_members]
    candidate_results = []
    for config, values in zip(CANDIDATES, inner_scores):
        value = macro_f1(train.y.iloc[inner_validation], np.asarray(LABELS)[values.argmax(axis=1)])
        candidate_results.append({**config, "inner_macro_f1": value})
        print(f"{config['name']}: {value:.6f}", flush=True)
    ranked = sorted(range(len(CANDIDATES)),
                    key=lambda index: candidate_results[index]["inner_macro_f1"], reverse=True)
    trials = []
    for size in (1, 2, 3):
        indices = ranked[:size]
        schemes = [np.ones(size) / size]
        if size > 1:
            values = np.asarray([candidate_results[index]["inner_macro_f1"] for index in indices])
            schemes.append(values / values.sum())
        for weights in schemes:
            for subtle, neutral in itertools.product((-.15, -.10, -.05, 0., .05),
                                                     (-.05, 0., .05, .10, .15)):
                offsets = [0., subtle, neutral]
                combined = ensemble_scores(inner_scores, indices, weights, offsets)
                value = macro_f1(train.y.iloc[inner_validation],
                                 np.asarray(LABELS)[combined.argmax(axis=1)])
                trials.append({"indices": indices, "weights": weights.tolist(),
                               "offsets": offsets, "inner_macro_f1": value})
    selected = max(trials, key=lambda item: (item["inner_macro_f1"], -len(item["indices"]),
                                            -sum(abs(x) for x in item["offsets"])))
    selected_configs = [CANDIDATES[index] for index in selected["indices"]]
    print(f"Selected {[c['name'] for c in selected_configs]}, weights={selected['weights']}, "
          f"offsets={selected['offsets']}, F1={selected['inner_macro_f1']:.6f}", flush=True)
    joblib.dump({"members": inner_members, "selection": selected},
                inner_path, compress=3)
    del inner_members, inner_scores
    gc.collect()

    outer_members = fit_members(selected_configs, texts.iloc[fit], train.y.iloc[fit])
    outer_scores = [decision(member, texts.iloc[holdout]) for member in outer_members]
    # Selection indices refer to the original candidate list; outer members are now compact.
    compact_indices = list(range(len(outer_members)))
    combined = ensemble_scores(outer_scores, compact_indices, selected["weights"], selected["offsets"])
    predictions = np.asarray(LABELS)[combined.argmax(axis=1)]
    value = macro_f1(train.y.iloc[holdout], predictions)
    print(f"Outer Macro F1: {value:.6f}; submission 3: {previous_score:.6f}", flush=True)
    pd.DataFrame({"id": train.id.iloc[holdout].to_numpy(),
                  "y": train.y.iloc[holdout].to_numpy(), "y_pred": predictions}).to_csv(
                      validation_path, index=False)
    metrics = {
        "model": "independent TF-IDF NB-SVM views",
        "run_name": name,
        "reference_validation": args.reference_validation,
        "candidates": candidate_results, "selection_trials": trials,
        "selected": selected, "selected_configs": selected_configs,
        "macro_f1": value, "submission3_macro_f1": previous_score,
        "improvement": value - previous_score,
        "classification_report": classification_report(train.y.iloc[holdout], predictions,
                                                        labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(train.y.iloc[holdout], predictions,
                                              labels=LABELS).tolist(),
        "split": {"outer_folds": 5, "outer_seed": 42, "inner_folds": 4, "inner_seed": 43,
                  "inner_fit": len(inner_fit), "inner_validation": len(inner_validation),
                  "outer_fit": len(fit), "outer_validation": len(holdout)},
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in paths.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "submission": None,
    }
    joblib.dump({"members": outer_members, "weights": selected["weights"],
                 "offsets": selected["offsets"]},
                validation_model_path, compress=3)
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    require(value > previous_score, "No validation improvement; submission 4 not generated")
    del outer_members, outer_scores, combined
    gc.collect()

    print("Refitting selected views on all labeled rows", flush=True)
    final_members = fit_members(selected_configs, texts, train.y)
    test_scores = [decision(member, test_texts) for member in final_members]
    combined = ensemble_scores(test_scores, list(range(len(final_members))),
                               selected["weights"], selected["offsets"])
    test_predictions = pd.Series(np.asarray(LABELS)[combined.argmax(axis=1)], index=test.id)
    submission = sample[["id"]].copy()
    submission["y_pred"] = submission.id.map(test_predictions)
    require(submission.id.equals(sample.id), "Submission order changed")
    require(submission.y_pred.notna().all() and submission.y_pred.isin(LABELS).all(),
            "Invalid predictions")
    require(not output.exists(), f"Preserving existing {output}")
    submission.to_csv(output, index=False, mode="x")
    bundle = {"members": final_members, "weights": selected["weights"],
              "offsets": selected["offsets"]}
    require(not model_path.exists(), f"Preserving existing {model_path}")
    joblib.dump(bundle, model_path, compress=3)
    reloaded = joblib.load(model_path)
    check_scores = [decision(member, test_texts) for member in reloaded["members"]]
    check = ensemble_scores(check_scores, list(range(len(check_scores))),
                            reloaded["weights"], reloaded["offsets"])
    require(np.array_equal(np.asarray(LABELS)[check.argmax(axis=1)],
                           submission.y_pred.to_numpy()), "Reload mismatch")
    metrics.update({"submission": str(output.relative_to(ROOT)), "test_rows": len(submission),
                    "model_reload_verified": True,
                    "elapsed_seconds": round(time.perf_counter() - started, 2)})
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {output.name}: {len(submission)} verified rows", flush=True)


if __name__ == "__main__":
    main()
