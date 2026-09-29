#!/usr/bin/env bash
# Reproduce submission 9 from the package root. Requires data/raw/{train,test,sample_submission}.csv.
# Stage 1 (GPU, optional): python src/train_submission5.py --device cuda
#   then replace recorded_xlmr_outputs with "." and add --allow-new-xlmr to stage 2.
set -euo pipefail
XLMR_DIR="${XLMR_DIR:-recorded_xlmr_outputs}"
EXTRA="${EXTRA:-}"
PY="${PYTHON:-python3}"
"$PY" src/train_submission7.py --xlmr-dir "$XLMR_DIR" $EXTRA   # NB-SVM views + blend (submission 7)
"$PY" src/nbsvm_oof_submission9.py                               # out-of-fold NB-SVM scores
"$PY" src/train_submission9.py --xlmr-dir "$XLMR_DIR"            # final stack -> submissions/submission9_robust_stack.csv
