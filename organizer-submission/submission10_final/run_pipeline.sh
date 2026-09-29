#!/usr/bin/env bash
# Reproduce submission 10 from the package root. Requires data/raw/{train,test,sample_submission}.csv.
# GPU stages (optional; recorded outputs are included):
#   python src/train_submission5.py --device cuda                  -> XLM-R-base      (then XLMR_DIR=. EXTRA=--allow-new-xlmr)
#   python src/train_submission6_xlmr_large.py --root large_run     -> XLM-R-large x2  (then LARGE_DIR=large_run; needs 2 GPUs)
set -euo pipefail
PY="${PYTHON:-python3}"
XLMR_DIR="${XLMR_DIR:-recorded_xlmr_outputs}"
LARGE_DIR="${LARGE_DIR:-recorded_xlmr_large_outputs}"
EXTRA="${EXTRA:-}"
"$PY" src/train_submission7.py --xlmr-dir "$XLMR_DIR" $EXTRA        # NB-SVM views + base blend
"$PY" src/nbsvm_oof_submission9.py                                    # out-of-fold NB-SVM scores
"$PY" src/train_submission9.py --run-name submission10 \
      --xlmr-dir "$XLMR_DIR" --large-dir "$LARGE_DIR"                 # final stack -> submissions/submission10_robust_stack.csv
