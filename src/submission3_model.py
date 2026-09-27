"""Feature extraction and inference for the third submission."""

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from nbsvm import NBSVM
from train_baseline import LABELS, SEED, normalize


META_NAMES = ["length", "words", "mean_word_length", "uppercase_ratio",
              "digit_ratio", "punctuation_ratio", "emoji_count", "bangla_ratio",
              "exclamations", "questions", "repeated_runs", "unique_word_ratio",
              "has_bangla", "mostly_uppercase", "whitespace_ratio"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
BANGLA = re.compile(r"[\u0980-\u09FF]")
PUNCT = re.compile(r"[!?.,;:@#$%^&*()_+\-=\[\]{}|\\/<>~`\"']")
REPEATS = re.compile(r"(.)\1{2,}")


def meta_features(texts):
    rows = []
    for text in texts:
        length = max(len(text), 1)
        words = text.split()
        upper = sum(c.isupper() for c in text)
        bangla = len(BANGLA.findall(text))
        rows.append([
            np.log1p(length), np.log1p(len(words)),
            np.mean([len(w) for w in words]) if words else 0,
            upper / length, sum(c.isdigit() for c in text) / length,
            len(PUNCT.findall(text)) / length, np.log1p(len(EMOJI.findall(text))),
            bangla / length, np.log1p(text.count("!")), np.log1p(text.count("?")),
            np.log1p(len(REPEATS.findall(text))),
            len(set(text.lower().split())) / max(len(words), 1),
            float(bangla > 0), float(upper > length * .5),
            sum(c.isspace() for c in text) / length,
        ])
    return np.asarray(rows, dtype=np.float32)


def majority_labels(texts, labels):
    """Change conflicting labels only when a group has a strict majority."""
    table = pd.DataFrame({"text": np.asarray(texts), "label": np.asarray(labels)})
    replacements = {}
    for text, group in table.groupby("text", sort=False):
        counts = group.label.value_counts()
        if len(counts) > 1 and counts.iloc[0] > len(group) / 2:
            replacements[text] = counts.index[0]
    return np.asarray([replacements.get(text, label)
                       for text, label in zip(table.text, table.label)])


def fit_member(config, X, meta, y):
    """NB ratios apply only to text features; numeric features stay separate."""
    if config["kind"] == "lr":
        combined = hstack([X, csr_matrix(meta)], format="csr")
        model = OneVsRestClassifier(LogisticRegression(
            C=config["C"], solver="liblinear", class_weight="balanced",
            max_iter=3000, random_state=SEED))
        model.fit(combined, y)
        return {"kind": "lr", "model": model, "config": config}
    if not config["meta"]:
        model = NBSVM(C=config["C"], alpha=config["alpha"], random_state=SEED).fit(X, y)
        return {"kind": "nb_plain", "model": model, "config": config}
    binary = X.copy()
    binary.data[:] = 1
    ratios, models = [], []
    for label in LABELS:
        target = np.asarray(y) == label
        p = np.asarray(binary[target].sum(axis=0)).ravel() + config["alpha"]
        q = np.asarray(binary[~target].sum(axis=0)).ravel() + config["alpha"]
        ratio = np.log((p / p.sum()) / (q / q.sum()))
        combined = hstack([X.multiply(ratio), csr_matrix(meta)], format="csr")
        model = LinearSVC(C=config["C"], class_weight="balanced", dual="auto",
                          max_iter=10000, random_state=SEED).fit(combined, target)
        ratios.append(ratio)
        models.append(model)
    return {"kind": "nb_meta", "ratios": ratios, "models": models, "config": config}


def member_scores(member, X, meta):
    if member["kind"] == "nb_meta":
        return np.column_stack([
            model.decision_function(hstack([X.multiply(ratio), csr_matrix(meta)], format="csr"))
            for ratio, model in zip(member["ratios"], member["models"])
        ])
    matrix = hstack([X, csr_matrix(meta)], format="csr") if member["kind"] == "lr" else X
    model = member["model"]
    assert list(model.classes_) == LABELS
    return model.decision_function(matrix)


def combine_scores(scores, weights, offsets):
    return sum(float(w) * s for w, s in zip(weights, scores)) + np.asarray(offsets)


def predict_bundle(bundle, raw_texts):
    texts = pd.Series(list(raw_texts))
    X = bundle["features"].transform(texts.map(normalize))
    meta = bundle["scaler"].transform(meta_features(texts)).astype(np.float32) * .2
    scores = [member_scores(member, X, meta) / scale
              for member, scale in zip(bundle["members"], bundle["scales"])]
    combined = combine_scores(scores, bundle["weights"], bundle["offsets"])
    return np.asarray(LABELS)[combined.argmax(axis=1)]
