# Competition data

Data is for authorized competition use only and is not stored on GitHub.

| File in `data/raw/` | Columns | Purpose |
| --- | --- | --- |
| `train.csv` | `id`, `text`, `y` | Labeled training posts |
| `test.csv` | `id`, `text` | Posts requiring predictions |
| `sample_submission.csv` | `id`, `y_pred` | Required submission layout |

The supplied ZIP contains these three files, totaling 6,753,862 uncompressed bytes. Headers were checked directly against the archive.

Local organization:

- `data/archives/`: original downloaded archive.
- `data/raw/`: original, unmodified extracted CSVs.
- `data/processed/`: optional generated splits or transformed data.

The `id` field identifies a post. The `text` field contains its content. Labels `y` and predictions `y_pred` use integers 0, 1, or 2. Follow the sample submission's IDs and format when generating predictions.

Get the files from the [official competition](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/data). All directories above are ignored by Git. Do not publish data excerpts through notebooks, logs, reports, or Git LFS.
