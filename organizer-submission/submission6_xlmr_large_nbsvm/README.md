# HerWILL × UAP Datathon 2026 — Submission 6

This package contains the complete source needed to train and evaluate the submission 6 classifier and generate a Kaggle-format prediction file. The model blends a fine-tuned XLM-R-large transformer with three TF-IDF NB-SVM models. Competition data and generated model weights are excluded.

## Files

- `src/train_submission6.py` — main script. Contains all preprocessing, grouped splits, XLM-R-large fine-tuning, the NB-SVM models, blend selection, validation and prediction. No other project modules are imported.
- `requirements.txt` — tabular and evaluation dependencies.
- `requirements-transformer.txt` — transformer dependencies; includes `requirements.txt`.
- `kaggle/submission6-script/kernel-metadata.json` — private Kaggle script configuration.

There are no separate helper or preprocessing files; everything is in the main script.

## Data

The script expects the original competition files without modification:

```text
data/raw/train.csv              id,text,y
data/raw/test.csv               id,text
data/raw/sample_submission.csv  id,y_pred
```

The input files are checked against SHA-256 hashes recorded in the script. On Kaggle, the script finds the attached competition data under `/kaggle/input` automatically. Only the provided competition data is used for training; no external datasets.

## Environment

Use Python 3.11 or newer and a CUDA GPU with at least 15 GB memory (a Kaggle T4 is sufficient). Install the pinned dependencies from the package root:

```bash
python -m pip install -r requirements-transformer.txt
```

The pretrained model is `FacebookAI/xlm-roberta-large` at revision `c23d21b0620b635a76227c604d44e43a9f0ee389`. Internet access is required for the initial download unless a local copy is supplied with `--pretrained <directory>`.

## Run

Place the three competition CSVs under `data/raw/`, then run from the package root:

```bash
python src/train_submission6.py --device cuda --run-name submission6
```

Runtime is roughly 2.5–3.5 hours on one T4. If GPU memory is short, add `--batch-size 8 --accumulation 4` (same effective batch size).

## Method

Preprocessing: Unicode NFC and whitespace normalization. XLM-R receives original-case text; the NB-SVM models and duplicate grouping use lowercased text. Normalized duplicate texts always stay in the same split.

Validation: a 5-fold stratified group split (seed 42) reserves the outer holdout (9,563 rows); a 4-fold stratified group split (seed 43) of the remaining rows gives inner training (28,690) and inner validation (9,564).

Models:

1. **XLM-R-large** — learning rate 1e-5, AdamW (weight decay 0.01), batch 16 × gradient accumulation 2, 3 epochs, 10% warmup then linear decay, max length 128, gradient clipping 1.0, CUDA mixed precision, seed 42. Loss is cross-entropy with square-root inverse-frequency class weights.
2. **NB-SVM views** — three TF-IDF feature sets (word 1–3 + char_wb 2–6 grams; word 1–2 + char 2–5 grams; char_wb 2–6 grams), each with a one-vs-rest LinearSVC on Naive Bayes log-count-ratio features (`C=0.1`, `alpha=1.0`, balanced class weights), averaged with equal weights. These run in a parallel CPU process.
3. **Blend** — `w · z(log_softmax(XLM-R logits)) + (1 − w) · z(NB-SVM scores) + [0, o1, o2]`, where `z` divides by the scored set's standard deviation. The weight `w` and class offsets `o1`, `o2` are chosen by grid search on inner validation only.

Steps performed by the default run:

1. Train XLM-R-large on inner training; the best inner-validation epoch sets the epoch count. The run stops if the model collapses (inner Macro F1 ≤ 0.40).
2. Choose the blend weight and offsets on inner validation.
3. Train fresh models on all non-holdout rows and evaluate the blend once on the outer holdout.
4. Continue only if the outer Macro F1 exceeds submission 5's 0.5590271875.
5. Train final models on all 47,817 labeled rows, reload the saved transformer to verify predictions, and write `submissions/submission6_xlmr_large_nbsvm.csv`.

Existing output paths are never overwritten. Use another `--run-name` for a separate run.

## Kaggle script

Attach the competition data, enable a GPU and internet access, then push from the package root:

```bash
kaggle kernels push -p kaggle/submission6-script
kaggle kernels status swarlok/submission-6-xlm-r-large-nb-svm
kaggle kernels output swarlok/submission-6-xlm-r-large-nb-svm -p kaggle-output
```

Change the `id` in `kernel-metadata.json` to your own Kaggle username to run it under another account.

## Expected outputs

- `reports/submission6_metrics.json` — outer Macro F1 of the blend, XLM-R-large alone and NB-SVM alone, the selected blend settings, and run metadata.
- `data/processed/submission6_validation.csv` and its logits file — outer-holdout predictions.
- `models/submission6/` — manifest, split indices, training histories, NB-SVM scores and selected checkpoints.
- `submissions/submission6_xlmr_large_nbsvm.csv` — `id,y_pred` file, written only after the validation gate passes.

The competition metric is Macro F1 across labels 0 (explicitly toxic), 1 (subtly toxic) and 2 (non-toxic).
