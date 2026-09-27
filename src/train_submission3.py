"""Train submission 3 with inner selection and a fixed outer comparison.

Run from any directory: python3 src/train_submission3.py
For a repeat that preserves existing results: --run-name submission3_repeat1
"""

import argparse
import gc
import hashlib
import itertools
import json
import platform
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from train_baseline import ROOT, LABELS, SEED, normalize, require
from train_submission2 import features
from submission3_model import (META_NAMES, meta_features, majority_labels, fit_member,
                               member_scores, combine_scores, predict_bundle)


CANDIDATES = [
    dict(name="submission2_reference", kind="nb", C=.25, alpha=1., meta=False, clean=False),
    dict(name="nb_meta_025", kind="nb", C=.25, alpha=1., meta=True, clean=False),
    dict(name="nb_meta_01", kind="nb", C=.1, alpha=1., meta=True, clean=False),
    dict(name="nb_meta_alpha05", kind="nb", C=.25, alpha=.5, meta=True, clean=False),
    dict(name="nb_meta_majority", kind="nb", C=.25, alpha=1., meta=True, clean=True),
    dict(name="lr_meta", kind="lr", C=4., alpha=1., meta=True, clean=False),
]


def score(y, pred):
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_stage(raw_texts, labels, configs):
    vectorizer = features()
    texts = raw_texts.map(normalize)
    X = vectorizer.fit_transform(texts)
    scaler = StandardScaler()
    meta = scaler.fit_transform(meta_features(raw_texts)).astype(np.float32) * .2
    members, scales, changes = [], [], []
    for config in configs:
        y = majority_labels(texts, labels) if config["clean"] else np.asarray(labels)
        member = fit_member(config, X, meta, y)
        # One scalar per model, learned from its fitting rows only. Not probabilities.
        scale = max(float(member_scores(member, X, meta).std()), 1e-8)
        members.append(member)
        scales.append(scale)
        changes.append(int((y != np.asarray(labels)).sum()))
        print(f"  Fitted {config['name']}; changed labels: {changes[-1]}", flush=True)
    return dict(features=vectorizer, scaler=scaler, members=members, scales=scales,
                label_changes=changes)


def score_stage(bundle, raw_texts):
    X = bundle["features"].transform(raw_texts.map(normalize))
    meta = bundle["scaler"].transform(meta_features(raw_texts)).astype(np.float32) * .2
    return [member_scores(member, X, meta) / scale
            for member, scale in zip(bundle["members"], bundle["scales"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission3")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Use letters, digits and underscores for run name")
    started = time.perf_counter()
    warnings.filterwarnings("error", category=ConvergenceWarning)
    name = args.run_name
    model_dir = ROOT / f"models/{name}"
    report_path = ROOT / f"reports/{name}_metrics.json"
    validation_path = ROOT / f"data/processed/{name}_validation.csv"
    submission_path = ROOT / f"submissions/{name}_ensemble.csv"
    for path in (model_dir, report_path, validation_path, submission_path):
        require(not path.exists(), f"Preserving existing {path}; choose a new --run-name")
    protected = [p for directory in ("data/raw", "data/archives", "models", "submissions")
                 for p in (ROOT / directory).rglob("*") if p.is_file()]
    old_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    paths = {n: ROOT / f"data/raw/{n}.csv" for n in ("train", "test", "sample_submission")}
    train, test, sample = [pd.read_csv(p, dtype={"id": str}, keep_default_na=False) for p in paths.values()]
    for frame, columns in ((train, ["id", "text", "y"]), (test, ["id", "text"]),
                           (sample, ["id", "y_pred"])):
        require(frame.columns.tolist() == columns, "Unexpected input schema")
        require(frame.id.is_unique and frame.id.ne("").all(), "Invalid IDs")
    require(set(train.y.unique()) == set(LABELS), "Invalid labels")
    require(set(test.id) == set(sample.id), "Test/sample ID mismatch")
    require(not set(train.id).intersection(test.id), "Train/test ID overlap")
    texts = train.text.map(normalize)
    require(texts.ne("").all() and test.text.map(normalize).ne("").all(), "Empty texts")
    groups = pd.factorize(texts, sort=True)[0]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit, holdout = next(outer.split(texts, train.y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
    a, b = next(inner.split(texts.iloc[fit], train.y.iloc[fit], groups[fit]))
    a, b = fit[a], fit[b]
    require(not set(groups[a]).intersection(groups[b]), "Inner group overlap")
    require(not set(groups[fit]).intersection(groups[holdout]), "Outer group overlap")
    previous = pd.read_csv(ROOT / "data/processed/submission2_validation.csv", dtype={"id": str})
    require(previous.id.tolist() == train.id.iloc[holdout].tolist(), "Submission 2 holdout differs")
    require(np.array_equal(previous.y, train.y.iloc[holdout]), "Holdout labels differ")
    prior_score = score(previous.y, previous.y_pred)

    model_dir.mkdir(parents=True)
    source_dir = model_dir / "source"
    source_dir.mkdir()
    sources = [ROOT / f"src/{file}" for file in ("train_submission3.py", "submission3_model.py",
               "train_submission2.py", "train_baseline.py", "nbsvm.py")] + [ROOT / "requirements.txt"]
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    print(f"Inner split: {len(a)} fit / {len(b)} validation. Outer holdout: {len(holdout)}.", flush=True)
    inner_bundle = fit_stage(train.text.iloc[a], train.y.iloc[a], CANDIDATES)
    inner_scores = score_stage(inner_bundle, train.text.iloc[b])
    results = [{**config, "inner_macro_f1": score(train.y.iloc[b], np.asarray(LABELS)[s.argmax(axis=1)])}
               for config, s in zip(CANDIDATES, inner_scores)]
    for row in results:
        print(f"  {row['name']}: {row['inner_macro_f1']:.6f}", flush=True)
    ranked = sorted(range(len(results)), key=lambda i: results[i]["inner_macro_f1"], reverse=True)
    trials = []
    # Small, fixed search. The outer holdout is never used to choose weights or offsets.
    for k in (1, 2, 3):
        indices = ranked[:k]
        schemes = [np.ones(k) / k]
        if k > 1:
            weights = np.asarray([results[i]["inner_macro_f1"] for i in indices])
            schemes.append(weights / weights.sum())
        for weights in schemes:
            for subtle, neutral in itertools.product((-.1, 0., .1), (0., .1, .2, .3)):
                offsets = [0., subtle, neutral]
                combined = combine_scores([inner_scores[i] for i in indices], weights, offsets)
                value = score(train.y.iloc[b], np.asarray(LABELS)[combined.argmax(axis=1)])
                trials.append(dict(indices=indices, weights=weights.tolist(), offsets=offsets,
                                   inner_macro_f1=value))
    best = max(trials, key=lambda row: (row["inner_macro_f1"], -len(row["indices"]),
                                      -sum(abs(v) for v in row["offsets"])))
    selected = [CANDIDATES[i] for i in best["indices"]]
    joblib.dump(inner_bundle, model_dir / "inner_candidates.joblib", compress=3)
    del inner_bundle, inner_scores
    gc.collect()
    print(f"Selected: {[c['name'] for c in selected]}, weights={best['weights']}, "
          f"offsets={best['offsets']}, inner F1={best['inner_macro_f1']:.6f}", flush=True)
    outer_bundle = fit_stage(train.text.iloc[fit], train.y.iloc[fit], selected)
    outer_bundle.update(weights=best["weights"], offsets=best["offsets"])
    pred = predict_bundle(outer_bundle, train.text.iloc[holdout])
    value = score(train.y.iloc[holdout], pred)
    print(f"Outer Macro F1: {value:.6f}; submission 2: {prior_score:.6f}", flush=True)
    joblib.dump(outer_bundle, model_dir / "validation_model.joblib", compress=3)
    pd.DataFrame(dict(id=train.id.iloc[holdout].to_numpy(), y=train.y.iloc[holdout].to_numpy(),
                      y_pred=pred)).to_csv(validation_path, index=False)
    metrics = dict(
        run_name=name, candidates=results, selection_trials=trials, selected=selected,
        ensemble=best, macro_f1=value, submission2_macro_f1=prior_score,
        improvement=value-prior_score, split=dict(outer_seed=42, outer_folds=5, fold=0,
            inner_seed=43, inner_folds=4, inner_fit=len(a), inner_validation=len(b),
            outer_fit=len(fit), outer_validation=len(holdout)),
        features=dict(word_ngrams=[1,2], char_ngrams=[2,5], word_max=150000, char_max=250000,
                      min_df=2, meta_names=META_NAMES, meta_weight=.2),
        validation_label_changes=outer_bundle["label_changes"],
        classification_report=classification_report(train.y.iloc[holdout], pred, labels=LABELS,
                                                     output_dict=True, zero_division=0),
        confusion_matrix=confusion_matrix(train.y.iloc[holdout], pred, labels=LABELS).tolist(),
        input_sha256={n:digest(p) for n,p in paths.items()},
        source_sha256={p.name:digest(p) for p in sources}, preserved_artifact_sha256=old_hashes,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
        submission=None)
    report_path.write_text(json.dumps(metrics, indent=2)+"\n")
    if value > prior_score:
        del outer_bundle
        gc.collect()
        print("Refitting on all labeled rows", flush=True)
        final = fit_stage(train.text, train.y, selected)
        final.update(weights=best["weights"], offsets=best["offsets"])
        test_pred = pd.Series(predict_bundle(final, test.text), index=test.id)
        submission = sample[["id"]].copy()
        submission["y_pred"] = submission.id.map(test_pred)
        require(submission.y_pred.notna().all() and submission.y_pred.isin(LABELS).all(), "Invalid predictions")
        require(submission.id.equals(sample.id), "Submission order changed")
        model_path = model_dir / "final_model.joblib"
        joblib.dump(final, model_path, compress=3)
        reloaded = joblib.load(model_path)
        require(np.array_equal(predict_bundle(reloaded, test.text), test_pred.to_numpy()), "Reload mismatch")
        submission.to_csv(submission_path, index=False)
        saved = pd.read_csv(submission_path, dtype={"id":str})
        require(saved.equals(submission), "CSV round-trip mismatch")
        metrics.update(submission=str(submission_path.relative_to(ROOT)), test_rows=len(saved),
                       final_label_changes=final["label_changes"], model_reload_verified=True)
        print(f"Saved and verified {submission_path.name}: {len(saved)} rows", flush=True)
    else:
        print("No outer improvement. Artifacts preserved; no submission generated.", flush=True)
    for path, checksum in old_hashes.items():
        require(digest(ROOT/path) == checksum, f"Earlier artifact changed: {path}")
    metrics.update(elapsed_seconds=round(time.perf_counter()-started,2), earlier_artifacts_unchanged=True)
    report_path.write_text(json.dumps(metrics, indent=2)+"\n")


if __name__ == "__main__":
    main()
