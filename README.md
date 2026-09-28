# HerWILL × UAP Datathon 2026

Research Expo 2.0 competition workspace for classifying harmful Bangla, English, and Banglish social media text.

Teammates: start with [setup and next-model instructions](docs/team-setup.md). This private repository includes the original CSVs, dependencies, training scripts, and results needed to start a new experiment.

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
data/           Original CSVs in raw/ and data documentation
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

The saved pipeline expects text normalized with `normalize()` in the training script before inference. The team's first submission achieved a public Macro F1 of **0.50248**.

## Second submission

The [second model](reports/submission2.md) uses NB-SVM and scores **0.52756 validation Macro F1**, compared with **0.51137** for the first model. Run `python3 src/train_submission2.py` after the baseline to reproduce it. Its prediction file is `submissions/submission2_tfidf_nbsvm.csv`.

After two entries, the team's best public Macro F1 was **0.53744**, up from **0.50248**.

## Third submission

The [third experiment](reports/submission3.md) achieved **0.53442 validation Macro F1**. It selected an NB-SVM with numerical text features, `C=0.1`, and a class-score adjustment using inner validation. It uses original labels and one selected model.

Run `python3 src/train_submission3.py` to reproduce training, or `python3 src/predict_submission3.py` to regenerate predictions from the saved model. Existing outputs are preserved; a training rerun requires a new `--run-name`. The upload file is `submissions/submission3_tfidf_nbsvm_meta.csv`, an identical copy of the retained pipeline output `submission3_ensemble.csv`.

Kaggle accepted submission 3 with a public Macro F1 of **0.54601**, improving by **0.00857** over submission 2. Private leaderboard results are pending.

## Fourth submission

The [fourth model](reports/submission4.md) combines three TF-IDF NB-SVM feature configurations. It achieved **0.53766 validation Macro F1** and **0.54759 public Macro F1**, improving the public score by **0.00158** over submission 3. The accepted file is `submissions/submission4_tfidf_nbsvm_views.csv`.

## Fifth submission

The [fifth model](reports/submission5.md) fine-tunes XLM-R-base with grouped duplicate-aware validation and class-weighted loss. It achieved **0.55903 outer validation Macro F1** and **0.56654 public Macro F1**, improving the public score by **0.01895** over submission 4. The accepted file is `submissions/submission5_xlmr.csv`.

## Sixth submission

The [sixth model](reports/submission6.md) blends XLM-R-large with the submission 4 NB-SVM views, choosing the blend weight and class offsets on inner validation. Launch it on a Kaggle T4 with `kaggle kernels push -p kaggle/submission6-script`. It writes `submissions/submission6_xlmr_large_nbsvm.csv` only if its outer Macro F1 beats submission 5.

## Data access

Download the files through the [competition Data page](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/data) after obtaining authorized access. Place the three CSV files in `data/raw/`. See [data/README.md](data/README.md) for their schemas.

The organizers restrict the dataset to this competition and prohibit sharing or publishing it outside the competition. The three original CSVs in `data/raw/` are tracked for authorized teammates in this private repository. Dataset archives, processed data, model artifacts, and submissions remain excluded from version control. Keep repository access limited to the competition team.

## Participation requirements

See [docs/competition.md](docs/competition.md) for the rules summary and submission requirements. Confirm the deadline and submission limits on Kaggle before submitting.

## License

The existing [MIT license](LICENSE) applies to repository code. It does not grant rights to the competition dataset, pretrained model weights, or other third-party materials.
