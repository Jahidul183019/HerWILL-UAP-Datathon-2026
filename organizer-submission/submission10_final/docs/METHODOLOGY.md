# Submission 10 methodology

## Problem and data

The task is to classify each post as 0 (explicitly toxic/hateful), 1 (subtly toxic) or 2 (non-toxic).

- **Train:** 47,817 labelled posts (49.7% / 38.0% / 12.3%).
- **Test:** 11,955 posts.
- **Languages:** about 42% of posts contain Bangla script; the rest are English or romanised Banglish.
- **Length:** posts are short (median 8 words, 99th percentile 58).

Only the competition files are used.

## Validation design

Every stage uses the same fixed split:

- **Duplicate groups:** texts are normalised (Unicode NFC, lowercase, collapsed whitespace) to form groups, so identical posts never appear on both sides of a split.
- **Outer split:** `StratifiedGroupKFold(5, seed 42)`; its first fold (9,563 rows) is the outer holdout.
- **Inner split:** `StratifiedGroupKFold(4, seed 43)` on the remaining 38,254 rows gives inner training (28,690) and inner validation (9,564).

For every base model, three versions are trained:

1. on inner training, to score inner validation;
2. on outer training, to score the outer holdout;
3. on all rows, to score the test set.

Every score the stacker sees is therefore out-of-sample. Settings are chosen on inner data. The outer holdout is scored once. The final stacker is also checked by repeated nested cross-validation.

## Base models

### XLM-R-base (stage 1a, `train_submission5.py`)

| Setting | Value |
| --- | --- |
| Checkpoint | `FacebookAI/xlm-roberta-base`, pinned revision |
| Input | Original-case text; NFC and whitespace normalisation; max length 128 |
| Optimiser | AdamW, learning rate 2e-5, weight decay 0.01, 10% warmup then linear decay |
| Batch | 8 × gradient accumulation 4 |
| Other | Gradient clipping 1.0, CUDA mixed precision, seed 42 |
| Loss | Cross-entropy with square-root inverse-frequency class weights |
| Epochs | 3, chosen on inner validation |

Outer holdout Macro F1: 0.5590.

### XLM-R-large (stage 1b, `train_submission6_xlmr_large.py`)

| Setting | Value |
| --- | --- |
| Checkpoint | `FacebookAI/xlm-roberta-large`, pinned revision |
| Input | Max length 192 |
| Optimiser | AdamW, learning rate 1e-5; weight decay 0.01, not applied to biases or LayerNorm; 10% warmup then linear decay |
| Batch | 4 × gradient accumulation 8 (effective 32) |
| Other | Gradient checkpointing, CUDA mixed precision |
| Loss | Same class-weighted loss as base |
| Seeds | 42 and 2026, trained in parallel on 2 × T4 |
| Epochs | 3, chosen from the average of both seeds' inner-validation probabilities |

Outer holdout Macro F1 by seed: 0.5796 (seed 42) and 0.5809 (seed 2026). An independent check on a second grouped fold, with its own epoch selection, gave 0.5804. These consistent results indicate no overfitting.

**Collapse guard.** The seed-42 model trained on all rows collapsed: its training loss stayed at about 0.91 and it predicts one class for every test post. Models trained on all rows have no validation data, so `train_submission9.py` checks each seed's test predictions and drops any seed that predicts one class for 90% or more of test posts. Submission 10 therefore uses seed 2026 for inner, outer and test consistently.

### TF-IDF NB-SVM views (stage 2, `train_submission7.py`)

Three feature views on lowercased text, all with `min_df=2` and sublinear TF:

1. word 1–3 grams + char_wb 2–6 grams;
2. word 1–2 grams + char 2–5 grams;
3. char_wb 2–6 grams.

Each view uses a one-vs-rest LinearSVC on Naive-Bayes log-count-ratio features (Wang & Manning, 2012; `C=0.1`, `alpha=1`, balanced class weights). Scores are divided by their standard deviation on the fitting rows, then averaged across views. Character n-grams capture Banglish spelling variation that subword models miss.

Outer holdout Macro F1: 0.5348.

## Key finding: ID order carries source information

Row IDs are not random:

- Consecutive IDs share the same script 98% of the time.
- Label rates change sharply between ID ranges.
- This suggests the dataset was compiled source by source (or thread by thread) before IDs were assigned.
- Test IDs are interleaved with training IDs across the whole range.

Predicting a post from only the labels of its nearest training rows by ID gives 0.61 Macro F1, more than any text model alone. We use this as a context feature alongside the text models. A row's own label is never used for its own features.

## Stacking features (stages 3–4, `train_submission9.py`)

For each row the stacker receives:

1. **Text-model scores:** XLM-R-base and XLM-R-large log-softmax scores, and NB-SVM scores. Each is divided by its overall standard deviation on the set being scored.
2. **ID-neighbour labels:**
   - label shares among the 1, 2, 5, 10, 25, 50 and 100 nearest *other* training rows by ID;
   - a distance-weighted share over the nearest 10;
   - log ID distance to the two nearest.
3. **Neighbour text scores:** mean NB-SVM scores of the 2, 10 and 50 nearest rows by ID among all train and test rows.
   - Train-row scores are out-of-fold (stage 3, five grouped folds); test-row scores come from the NB-SVM fitted on all training data.
   - This lets unlabelled neighbouring posts contribute.
4. **Post signals:** contains Bangla script, contains Latin script, log character length, and the Bangla share among the 10 nearest rows.

Leakage controls:

- No row uses its own label.
- Every text score is out-of-sample.
- The neighbour pool for training rows is "all other training rows", the same density test rows see.

## Stacker

- **Training rows:** the 19,127 rows with out-of-sample transformer scores (inner validation plus outer holdout).
- **Models:** multinomial logistic regression (`C=1`) and a histogram gradient-boosting classifier (learning rate 0.03, 15 leaves, 300 iterations, L2 1.0, minimum 50 samples per leaf). Their log-probabilities are averaged.
- **Class offsets:** `[0, o1, o2]` is added to the averaged log-probabilities to maximise Macro F1. Offsets are tuned by 3-fold cross-validation on the training rows over a grid from −1 to 1 in steps of 0.1. Final offsets: `[0, 0.2, 0.7]`.

## Results

| Model | Outer holdout | Kaggle public |
| --- | ---: | ---: |
| TF-IDF + LinearSVC (submission 1) | 0.5114 | 0.50248 |
| NB-SVM views (submission 4) | 0.5377 | 0.54759 |
| XLM-R-base (submission 5) | 0.5590 | 0.56654 |
| XLM-R-base + NB-SVM blend (submission 7) | 0.5798 | 0.59162 |
| + ID-neighbour labels, logistic regression stack (submission 8) | 0.6803 | 0.69770 |
| + 19k-row stacker, neighbour text scores, LR + gradient boosting (submission 9) | 0.6919 | 0.70190 |
| **+ XLM-R-large (submission 10, this package)** | **0.7057** | **0.71624** |

Submission 10 per-class F1 on the outer holdout: explicit 0.795, subtle 0.664, neutral 0.658.

**Robustness.** Repeated nested 5-fold CV over the 19,127 stacking rows, with offsets re-tuned inside each training fold, gave 0.7017, 0.6986 and 0.7011 (mean 0.7004, std 0.0013). Across submissions 8–10, each validation gain matched the public-leaderboard gain (+0.014 outer vs +0.014 public from submission 9 to 10). This indicates the stacker is not overfitting its tuning data.

## Limitations

- **ID ordering:** the ID-neighbour gain depends on the test set sharing the training set's ID ordering. That holds across the whole test ID range, so it should apply equally to the private leaderboard. It reflects how the dataset was assembled rather than the text alone. The text-only pipeline (submission 7, public 0.59162) is the content-only baseline.
- **Reused holdout:** the outer holdout informed several earlier experiments, so it is a development estimate. The nested CV provides a second estimate.
- **Reproducibility:** GPU training is not bit-for-bit deterministic. The package includes the recorded scores, so stages 2–4 can regenerate the submission on CPU.
