# HerWILL × UAP Datathon 2026

Research Expo 2.0 competition workspace for classifying harmful Bangla, English, and Banglish social media text.

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

No model has been trained and no predictions have been submitted yet.

## Data access

Download the files through the [competition Data page](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/data) after obtaining authorized access. Place the three CSV files in `data/raw/`. See [data/README.md](data/README.md) for their schemas.

The organizers restrict the dataset to this competition and prohibit sharing or publishing it outside the competition. Dataset archives, CSVs, processed data, model artifacts, and submissions are excluded from version control. Do not include dataset text in committed notebook outputs or reports.

## Participation requirements

See [docs/competition.md](docs/competition.md) for the supplied rules summary and submission requirements. Confirm the deadline and submission limits on Kaggle before submitting.

## License

The existing [MIT license](LICENSE) applies to repository code. It does not grant rights to the competition dataset, pretrained model weights, or other third-party materials.
