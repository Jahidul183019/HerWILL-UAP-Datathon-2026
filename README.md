# HerWILL × UAP Datathon 2026

Research Expo 2.0 competition workspace for classifying harmful Bangla, English, and Banglish social media text.

See [project status](docs/project-context.md) for progress and results. The public leaderboard uses approximately 40% of test data; final leaderboard results use the other 60%.

## Task

Predict one class per test post. Evaluation uses **Macro F1**.

| Label | Meaning |
| --- | --- |
| 0 | Explicitly toxic / hateful |
| 1 | Subtly toxic |
| 2 | Non-toxic / neutral |

## Repository layout

```text
data/           Data documentation; local datasets are ignored by Git
docs/           Competition requirements and references
notebooks/      Analysis and training notebooks
src/            Reusable preprocessing, training, and inference code
reports/        Aggregate results and methodology documentation
submissions/    Local prediction CSVs; ignored by Git
models/         Local trained models; ignored by Git
```

## Train the first baseline

The baseline combines word (1–2 gram) and character (3–5 gram) TF-IDF features with a balanced linear support vector classifier. It uses only the competition training data.

With Python 3.11 or newer, install the pinned dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 src/train_baseline.py
```

The script reserves the first fold of a seeded, five-fold stratified group split for validation. Normalized duplicate texts stay together, and feature vocabularies are fitted only on the training partition during validation. It then fits a fresh model on all labeled rows and generates test predictions in sample-submission order.

Outputs:

- `reports/baseline_metrics.json`: aggregate validation metrics, configuration, data checksums, and environment versions.
- `data/processed/baseline_validation.csv`: local validation predictions.
- `models/baseline_tfidf_svc.joblib`: local model fitted on all training rows.
- `submissions/submission1_tfidf_svc.csv`: local Kaggle-format predictions.

The saved pipeline expects text normalized with `normalize()` in the training script before inference. The team's first submission achieved a public Macro F1 of **0.50248**, placing second when recorded.

## Second submission

The [second model](reports/submission2.md) uses NB-SVM and scores **0.52756 validation Macro F1**, compared with **0.51137** for the first model. Run `python3 src/train_submission2.py` after the baseline to reproduce it. Its prediction file is `submissions/submission2_tfidf_nbsvm.csv`.

After two entries, the team's best public Macro F1 is **0.53744**, up from **0.50248**. The last recorded rank is second; private leaderboard results are pending.

## Data access

Download the files through the [competition Data page](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/data) after obtaining authorized access. Place the three CSV files in `data/raw/`. See [data/README.md](data/README.md) for their schemas.

The organizers restrict the dataset to this competition and prohibit sharing or publishing it outside the competition. Dataset archives, CSVs, processed data, model artifacts, and submissions are excluded from version control. Do not include dataset text in committed notebook outputs or reports.

## Participation requirements

See [docs/competition.md](docs/competition.md) for the rules summary and submission requirements. Confirm the deadline and submission limits on Kaggle before submitting.

## License

The existing [MIT license](LICENSE) applies to repository code. It does not grant rights to the competition dataset, pretrained model weights, or other third-party materials.
