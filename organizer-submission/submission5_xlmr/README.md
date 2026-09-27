# HerWILL × UAP Datathon 2026 — Submission 5

This package contains the complete source needed to train and evaluate the XLM-R classifier and generate a Kaggle-format prediction file. Competition data and generated model weights are excluded.

## Files

- `src/train_submission5.py` — main training, validation and prediction script. All preprocessing and split logic are included in this file.
- `src/benchmark_xlmr.py` — optional GPU benchmark and pretrained-checkpoint download helper.
- `src/run_submission5_local.py` — optional background launcher for macOS MPS.
- `requirements.txt` — tabular and evaluation dependencies.
- `requirements-transformer.txt` — transformer dependencies; includes `requirements.txt`.
- `kaggle/submission5-script/kernel-metadata.json` — private Kaggle script configuration.

## Data

The script expects the original competition files without modification:

```text
data/raw/train.csv              id,text,y
data/raw/test.csv               id,text
data/raw/sample_submission.csv  id,y_pred
```

The input files are checked against the SHA-256 hashes recorded in the script. On Kaggle, the script searches `/kaggle/input` for the attached competition data automatically. Do not include the competition CSVs when submitting this code package unless the organizers explicitly request them.

## Environment

Use Python 3.11 or newer and a CUDA GPU. Install the pinned dependencies from the package root:

```bash
python -m pip install -r requirements-transformer.txt
```

The model is `FacebookAI/xlm-roberta-base` at the exact revision pinned in the main script. Internet access is required for the initial model download unless a local copy is supplied with `--pretrained`.

## Run

Place the three competition CSVs under `data/raw/`, then run from the package root:

```bash
python src/train_submission5.py --device cuda --run-name submission5
```

The default run performs these steps:

1. Selects the epoch count on an inner stratified group split.
2. Trains a fresh model and evaluates it on the fixed outer holdout.
3. Continues only if outer Macro F1 exceeds the recorded submission 4 reference.
4. Trains a fresh final model on all labeled rows.
5. Reloads the saved model, verifies predictions and writes `submissions/submission5_xlmr.csv`.

Normalized duplicate texts remain in the same split. Training uses original-case text, square-root inverse-frequency class weights, AdamW, dropout, gradient clipping and mixed precision on CUDA. The run records input hashes, the copied training source, dependency versions, split indices, histories, predictions and checkpoint-retention details.

Existing output paths are never overwritten. Use another `--run-name` for a separate run.

## Kaggle script

Attach the private competition data, enable a GPU and internet access, then push the included script configuration from the package root:

```bash
kaggle kernels push -p kaggle/submission5-script
```

Check and download results with:

```bash
kaggle kernels status jahidul183019/submission-5-xlm-r
kaggle kernels output jahidul183019/submission-5-xlm-r -p kaggle-output
```

The Kaggle job is a Python script, not a notebook. Prediction submission to the competition remains a separate manual step.

## Optional local benchmark

```bash
python src/benchmark_xlmr.py --output models/submission5_benchmark
```

The macOS launcher uses MPS and `caffeinate`; it is not required for Kaggle or Linux execution.

## Expected outputs

- `reports/submission5_metrics.json` — outer validation metrics and run metadata.
- `data/processed/submission5_validation.csv` and its logits file — outer-holdout predictions.
- `models/submission5/` — manifest, split indices, histories and selected checkpoints.
- `submissions/submission5_xlmr.csv` — `id,y_pred` file generated only after the validation gate passes.

The competition score is Macro F1 across labels 0, 1 and 2.
