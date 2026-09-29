"""Out-of-fold NB-SVM scores for every training row (submission 9 neighbour features).

Uses the same five grouped folds as the outer split of submissions 1–8, so each row is
scored by views fitted without it or any duplicate of its text. Output:
models/submission9/nbsvm_oof.npy (train order). Test scores reuse submission 7's
models/submission7/nbsvm_final.npy (fitted on all rows).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import train_submission7 as s7


def main():
    out = s7.ROOT / "models/submission9/nbsvm_oof.npy"
    s7.require(not out.exists(), f"Preserving {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(s7.ROOT / "data/raw/train.csv", dtype={"id": str}, keep_default_na=False)
    texts = train.text.map(s7.normalize)
    groups = pd.factorize(texts, sort=True)[0]
    oof = np.zeros((len(train), 3))
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (fit, held) in enumerate(folds.split(texts, train.y, groups)):
        oof[held] = s7.nbsvm_scores(texts.iloc[fit], train.y.iloc[fit], texts.iloc[held])
        print(f"fold {fold} done", flush=True)
    np.save(out, oof)


if __name__ == "__main__":
    main()
