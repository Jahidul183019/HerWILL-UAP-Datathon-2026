# HerWILL × UAP Datathon 2026 — Submission 9 (best result)

Team: DU_NameDiyeKiAsheJay

Three-class harmful-text classification (0 explicitly toxic, 1 subtly toxic, 2 non-toxic) for Bangla, English and Banglish social media posts. Metric: Macro F1.

| Result | Macro F1 |
| --- | ---: |
| Kaggle public leaderboard | **0.70190** |
| Outer holdout (9,563 rows, never used for fitting) | 0.69187 |
| Repeated nested 5-fold cross-validation (19,127 rows × 3 repeats) | 0.6903 ± 0.0010 |

The method is described in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Package contents

| Path | Purpose |
| --- | --- |
| `src/train_submission5.py` | Stage 1 (GPU). Fine-tunes XLM-R-base; writes inner-validation, outer-holdout and test logits. |
| `src/train_submission7.py` | Stage 2 (CPU). Three TF-IDF NB-SVM views on the same splits; shared helpers (normalisation, grouped splits, NB-SVM, blending, metrics). |
| `src/nbsvm_oof_submission9.py` | Stage 3 (CPU). Out-of-fold NB-SVM scores for every training row. |
| `src/train_submission9.py` | Stage 4 (CPU). Builds the stacking features, validates, and writes the final CSV. **Main script.** |
| `run_pipeline.sh` | Runs stages 2–4 in order. |
| `recorded_xlmr_outputs/` | The exact XLM-R-base logits produced by our Kaggle GPU run of stage 1, with its metrics and prediction files (no model weights, no post text). Lets stages 2–4 reproduce the submitted file without a GPU. |
| `kaggle/submission5-script/kernel-metadata.json` | Kaggle script configuration used for stage 1 (T4 GPU). |
| `requirements.txt` | CPU dependencies (stages 2–4). |
| `requirements-transformer.txt` | Adds the transformer dependencies for stage 1. |
| `docs/METHODOLOGY.md` | Preprocessing, models, hyperparameters, validation and results. |
| `MANIFEST.sha256` | Checksums of every file in this package. |

All preprocessing and helper code is inside `src/`; there are no other modules.

## Data

Place the original competition files, unmodified, in `data/raw/` at the package root:

```text
data/raw/train.csv              id,text,y
data/raw/test.csv               id,text
data/raw/sample_submission.csv  id,y_pred
```

The scripts verify the SHA-256 of each file. Only the provided competition data is used; there are no external datasets. The only pretrained model is `FacebookAI/xlm-roberta-base` (revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`).

## Setup

Python 3.11 or newer. The pipeline script uses `python3`; set `PYTHON=/path/to/python` to override.

```bash
python -m pip install -r requirements.txt               # stages 2–4 (CPU)
python -m pip install -r requirements-transformer.txt   # also needed for stage 1 (GPU)
```

## Reproduce the submitted predictions (CPU, about 15 minutes)

From the package root:

```bash
./run_pipeline.sh
```

Output: `submissions/submission9_robust_stack.csv` (11,955 rows, `id,y_pred`, sample-submission order). The file we submitted to Kaggle has SHA-256 `a3eed1b6bea488c54573c17ba05ccad8a9023178fdd6d97c77475d93a2f8bd10`; it was produced with scikit-learn 1.9.0 for stages 3–4, so other library versions may change a small number of labels. Metrics are written to `reports/submission9_metrics.json`.

## Full rerun including XLM-R training (GPU, about 1 hour on a T4)

```bash
python src/train_submission5.py --device cuda            # stage 1, writes to ./models, ./reports, ./data/processed
XLMR_DIR=. EXTRA=--allow-new-xlmr ./run_pipeline.sh     # stages 2–4 on the fresh logits
```

On Kaggle, push the stage 1 script with the competition data attached (GPU and internet enabled):

```bash
kaggle kernels push -p kaggle/submission5-script
```

GPU training is not bit-for-bit deterministic, so a fresh stage 1 run gives slightly different logits and final predictions. `--allow-new-xlmr` turns off the checks that pin stage 2 to our recorded run.

## Outputs of each stage

| Stage | Writes |
| --- | --- |
| 1 | `models/submission5/`, `reports/submission5_metrics.json`, `data/processed/submission5_validation.*`, `submissions/submission5_xlmr.csv` |
| 2 | `models/submission7/nbsvm_{inner,outer,final}.npy`, `reports/submission7_metrics.json`, `submissions/submission7_xlmr_base_nbsvm.csv` |
| 3 | `models/submission9/nbsvm_oof.npy` |
| 4 | `reports/submission9_metrics.json`, `submissions/submission9_robust_stack.csv` |

Scripts never overwrite existing outputs; use a new `--run-name` (stages 2 and 4) or delete the output to rerun.
