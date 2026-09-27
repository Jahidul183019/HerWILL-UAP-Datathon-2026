# Team setup

The private repository includes the three original competition CSVs, training code for submissions 1–4, pinned dependencies, aggregate metrics, and experiment reports. Access is for authorized competition teammates.

## Install

```bash
git clone https://github.com/Jahidul183019/HerWILL-UAP-Datathon-2026.git
cd HerWILL-UAP-Datathon-2026
git switch -c team/next-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded runs used Python 3.14.3. The pinned packages require Python 3.11 or newer. These four models run on CPU; no Kaggle authentication, GPU, pretrained weights, or external datasets are required. GitHub access to the private repository is required to clone it.

## Inputs and results

| File | Rows | Columns |
| --- | ---: | --- |
| `data/raw/train.csv` | 47,817 | `id,text,y` |
| `data/raw/test.csv` | 11,955 | `id,text` |
| `data/raw/sample_submission.csv` | 11,955 | `id,y_pred` |

Labels are 0 (explicitly toxic/hateful), 1 (subtly toxic), and 2 (neutral). Input SHA-256 hashes are recorded in the metrics JSON files. The original archive, fitted models, validation predictions, and submission CSVs are generated/local artifacts and are not included in the clone. No existing local artifacts have been removed.

| Submission | Model | Validation Macro F1 | Public Macro F1 |
| --- | --- | ---: | ---: |
| 1 | TF-IDF + LinearSVC | 0.51137 | 0.50248 |
| 2 | TF-IDF + NB-SVM | 0.52756 | 0.53744 |
| 3 | NB-SVM + numerical text features | 0.53442 | 0.54601 |
| 4 | Three TF-IDF NB-SVM feature configurations | 0.53766 | 0.54759 |

## Start the next model

Create a new script such as `src/train_submission5.py`. Preserve scripts and artifacts for earlier submissions. Useful imports are `normalize`, `ROOT`, `LABELS`, and `SEED` from `train_baseline.py`; `NBSVM` from `nbsvm.py`; and the feature/ensemble functions in `train_submission4.py`.

To compare directly with existing experiments, reconstruct the established split:

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from train_baseline import normalize

texts = train.text.map(normalize)
groups = pd.factorize(texts, sort=True)[0]
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit, holdout = next(outer.split(texts, train.y, groups))

inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
a, b = next(inner.split(texts.iloc[fit], train.y.iloc[fit], groups[fit]))
inner_fit, inner_validation = fit[a], fit[b]
```

Expected sizes: outer fit 38,254, outer holdout 9,563, inner fit 28,690, inner validation 9,564. Fit feature extraction only on fitting rows. Choose settings on inner validation, then evaluate the selected configuration on the outer holdout using original labels. Keep normalized duplicate text groups together.

This holdout has already informed several experiments. Broader grouped validation is useful before treating small score gains as reliable. The public leaderboard covers approximately 40% of test data; final evaluation uses the other 60%. Track our scores and submission status, not leaderboard positions.

Use a new prefix for every experiment's models, predictions, and report. Produce `id,y_pred` in sample-submission order, save the model and preprocessing, and reload it to verify predictions. Kaggle uploads are handled manually.

## Reproduce earlier models from a fresh clone

Run submissions 1 and 2 once to create the reference validation files. Their scripts use fixed output paths, so keep copies of the committed metrics first:

```bash
mkdir -p data/archives/reference-metrics
cp reports/baseline_metrics.json reports/submission2_metrics.json data/archives/reference-metrics/
python src/train_baseline.py
python src/train_submission2.py
```

Reproduce submissions 3 and 4 with new output names to retain the committed reports:

```bash
python src/train_submission3.py --run-name reproduce3
python src/train_submission4.py --run-name reproduce4 --reference-validation data/processed/reproduce3_validation.csv
```

The last command creates `submissions/reproduce4_tfidf_nbsvm_views.csv`, the corresponding model files, and `reports/reproduce4_metrics.json`. If any selected output path already exists, choose another run name. The submission 4 search writes validation metrics even when the candidate fails to improve; it generates the final CSV only after improvement.

Training scripts for submissions 2–4 import other files in `src/`; keep the full directory when reproducing a run. For final organizer delivery, bundle dependencies or prepare a self-contained notebook/script rather than uploading one dependent file alone.

Before pushing new work, review `git status` and the staged file list. Keep archives, generated predictions, and trained models ignored; do not change repository visibility or redistribute competition data outside the team.
