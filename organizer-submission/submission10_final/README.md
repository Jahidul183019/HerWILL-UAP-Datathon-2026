# HerWILL × UAP Datathon 2026 — Final code submission (Submission 10)

Team: DU_NameDiyeKiAsheJay

Three-class harmful-text classification (0 explicitly toxic, 1 subtly toxic, 2 non-toxic) for Bangla, English and Banglish social media posts. Metric: Macro F1.

| Result | Macro F1 |
| --- | ---: |
| Kaggle public leaderboard | **0.71624** |
| Outer holdout (9,563 rows, never used for fitting) | 0.7057 |
| Repeated nested 5-fold cross-validation (19,127 rows × 3 repeats) | 0.7004 ± 0.0013 |

Submitted file: `submission10_robust_stack.csv`, SHA-256 `7194551e0a20696dd41f3a2076e55e1659bc78b5cbe5ddf433e938a6389ae80f`.

The full method is described in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Pipeline overview

```text
Stage 1a (GPU)  train_submission5.py            XLM-R-base        ─┐
Stage 1b (GPU)  train_submission6_xlmr_large.py XLM-R-large        ─┤ out-of-sample scores
Stage 2  (CPU)  train_submission7.py            TF-IDF NB-SVM      ─┤ for inner validation,
Stage 3  (CPU)  nbsvm_oof_submission9.py        NB-SVM out-of-fold ─┤ outer holdout and test
Stage 4  (CPU)  train_submission9.py            stacker (LR + gradient boosting) → final CSV
```

## Package contents

| Path | Purpose |
| --- | --- |
| `src/train_submission9.py` | **Main script** (stage 4). Builds the stacking features, validates (outer holdout plus repeated nested CV), and writes the final CSV. |
| `src/train_submission5.py` | Stage 1a. Fine-tunes XLM-R-base; writes inner-validation, outer-holdout and test logits. |
| `src/train_submission6_xlmr_large.py` | Stage 1b. Fine-tunes XLM-R-large with two seeds on two GPUs; writes the same logits per seed. |
| `src/train_submission7.py` | Stage 2. Three TF-IDF NB-SVM views on the same splits, plus shared helpers (normalisation, grouped splits, NB-SVM, metrics). |
| `src/nbsvm_oof_submission9.py` | Stage 3. Out-of-fold NB-SVM scores for every training row. |
| `run_pipeline.sh` | Runs stages 2–4 in order. |
| `recorded_xlmr_outputs/` | Scores from our Kaggle GPU run of stage 1a (logits, metrics, holdout predictions; no model weights or post text). |
| `recorded_xlmr_large_outputs/` | Scores from our Kaggle GPU run of stage 1b (per-seed logits, metrics, epoch selection). |
| `kaggle/submission5-script/`, `kaggle/submission6-xlmr-large/` | Kaggle script configurations used for stages 1a (T4) and 1b (2 × T4). |
| `requirements.txt` | CPU dependencies (stages 2–4). |
| `requirements-transformer.txt` | Adds transformer dependencies for stage 1. |
| `docs/METHODOLOGY.md` | Preprocessing, models, hyperparameters, validation, results and limitations. |
| `MANIFEST.sha256` | Checksums of every file in this package. |

All preprocessing and helper code is in `src/`; there are no other modules.

## Data

Place the original competition files, unmodified, in `data/raw/` at the package root:

```text
data/raw/train.csv              id,text,y
data/raw/test.csv               id,text
data/raw/sample_submission.csv  id,y_pred
```

The scripts check each file's SHA-256. Only the provided competition data is used, with no external datasets. The pretrained models are `FacebookAI/xlm-roberta-base` (revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`) and `FacebookAI/xlm-roberta-large` (revision `c23d21b0620b635a76227c604d44e43a9f0ee389`).

## Setup

Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt               # stages 2–4 (CPU)
python -m pip install -r requirements-transformer.txt   # also needed for stage 1 (GPU)
```

## Reproduce the submitted predictions (CPU only, about 15 minutes)

From the package root:

```bash
./run_pipeline.sh
```

This uses the recorded GPU outputs and writes `submissions/submission10_robust_stack.csv` (11,955 rows, `id,y_pred`, sample-submission order) and `reports/submission10_metrics.json`. The submitted file was produced with scikit-learn 1.9.0 for stages 3–4; other versions may change a few labels. The script uses `python3` by default; set `PYTHON=/path/to/python` to override.

## Full rerun including transformer training

Stage 1a (one T4, about 1 hour):

```bash
python src/train_submission5.py --device cuda
```

Stage 1b (two T4 GPUs, about 6 hours) into a separate directory so it does not collide with other outputs:

```bash
python src/train_submission6_xlmr_large.py --root large_run
```

Then run stages 2–4 on the fresh scores:

```bash
XLMR_DIR=. LARGE_DIR=large_run EXTRA=--allow-new-xlmr ./run_pipeline.sh
```

On Kaggle, attach the competition data, enable GPU and internet access, and push each configuration:

```bash
kaggle kernels push -p kaggle/submission5-script
kaggle kernels push -p kaggle/submission6-xlmr-large
```

Change the `id` in each `kernel-metadata.json` to your own Kaggle username. GPU training is not bit-for-bit deterministic, so a fresh stage 1 gives slightly different scores and final labels. `--allow-new-xlmr` turns off the checks that pin stage 2 to our recorded run.

## Outputs of each stage

| Stage | Writes |
| --- | --- |
| 1a | `models/submission5/`, `reports/submission5_metrics.json`, `data/processed/submission5_validation.*`, `submissions/submission5_xlmr.csv` |
| 1b | `<root>/models/submission6/seed{42,2026}/`, `<root>/reports/submission6_metrics.json` |
| 2 | `models/submission7/nbsvm_{inner,outer,final}.npy`, `reports/submission7_metrics.json` |
| 3 | `models/submission9/nbsvm_oof.npy` |
| 4 | `reports/submission10_metrics.json`, `submissions/submission10_robust_stack.csv` |

Scripts never overwrite existing outputs. Use a new `--run-name` or remove the output to rerun.
