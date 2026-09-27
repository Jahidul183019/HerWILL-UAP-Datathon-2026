"""Regenerate submission 3 from a saved bundle without retraining."""

import argparse

import joblib
import pandas as pd

from train_baseline import ROOT, LABELS, require
from submission3_model import predict_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="submission3")
    args = parser.parse_args()
    require(args.run_name.replace("_", "").isalnum(), "Invalid run name")
    output = ROOT / f"submissions/{args.run_name}_ensemble_reproduced.csv"
    require(not output.exists(), f"Preserving existing file: {output}")
    test = pd.read_csv(ROOT / "data/raw/test.csv", dtype={"id": str}, keep_default_na=False)
    sample = pd.read_csv(ROOT / "data/raw/sample_submission.csv", dtype={"id": str})
    require(test.id.is_unique and sample.id.is_unique and set(test.id) == set(sample.id), "Invalid IDs")
    bundle = joblib.load(ROOT / f"models/{args.run_name}/final_model.joblib")
    prediction = pd.Series(predict_bundle(bundle, test.text), index=test.id)
    result = sample[["id"]].copy()
    result["y_pred"] = result.id.map(prediction)
    require(result.y_pred.notna().all() and result.y_pred.isin(LABELS).all(), "Invalid predictions")
    original = ROOT / f"submissions/{args.run_name}_ensemble.csv"
    if original.exists():
        require(result.equals(pd.read_csv(original, dtype={"id": str})), "Predictions differ from original CSV")
    result.to_csv(output, index=False, mode="x")
    print(f"Verified {len(result):,} predictions; saved {output.name}")


if __name__ == "__main__":
    main()
