# Submission 9 methodology

## Problem and data

Classify each post as 0 (explicitly toxic/hateful), 1 (subtly toxic) or 2 (non-toxic). Train: 47,817 labelled posts (49.7% / 38.0% / 12.3%). Test: 11,955 posts. About 42% of posts contain Bangla script; the rest are English or romanised Banglish. Posts are short (median 8 words, 99th percentile 58). Only the competition files are used.

## Validation design

Every stage shares one split, fixed before any modelling:

- Texts are normalised (Unicode NFC, lowercase, collapsed whitespace) to form **duplicate groups**, so identical posts never fall on both sides of a split.
- **Outer split:** `StratifiedGroupKFold(5, seed 42)`; the first fold (9,563 rows) is the outer holdout.
- **Inner split:** `StratifiedGroupKFold(4, seed 43)` of the remaining 38,254 rows gives inner training (28,690) and inner validation (9,564).

Model selection happens on inner validation. The outer holdout is scored once per experiment. The final stacker is additionally checked by repeated nested cross-validation (below).

## Base models

### XLM-R-base (stage 1, `train_submission5.py`)

- `FacebookAI/xlm-roberta-base`, pinned revision; original-case text with NFC and whitespace normalisation; max length 128.
- AdamW, learning rate 2e-5, weight decay 0.01, 10% linear warmup then linear decay, batch 8 × gradient accumulation 4, gradient clipping 1.0, CUDA mixed precision, seed 42.
- Cross-entropy with square-root inverse-frequency class weights.
- The epoch count (3) is chosen on inner validation. Three models are trained: inner (for inner-validation logits), outer-fit (for outer-holdout logits) and all rows (for test logits).
- Outer holdout Macro F1: 0.5590. Kaggle public: 0.56654.

### TF-IDF NB-SVM views (stage 2, `train_submission7.py`)

Three feature views on lowercased text: word 1–3 grams + char_wb 2–6 grams; word 1–2 grams + char 2–5 grams; char_wb 2–6 grams (`min_df=2`, sublinear TF). Each view uses a one-vs-rest LinearSVC on Naive-Bayes log-count-ratio-scaled features (Wang & Manning, 2012; `C=0.1`, `alpha=1`, balanced class weights). Scores are divided by their standard deviation on the fitting rows and averaged. Character n-grams capture spelling variation in Banglish that subword models miss. Outer holdout Macro F1: 0.5348.

Blending these with XLM-R-base (submission 7) gave 0.5798 outer and 0.59162 public.

## Key finding: ID order carries source information

Row IDs are not random. Consecutive IDs share the same script 98% of the time, and label rates swing strongly between ID ranges. This suggests the dataset was compiled source by source or thread by thread before IDs were assigned. Test IDs are interleaved with training IDs across the whole range. Predicting a row from only the labels of its nearest training rows by ID gives 0.61 Macro F1, more than either text model alone. We use this as a context feature alongside the text models. It never uses a row's own label.

## Stacking features (stage 3–4)

For each row the stacker receives:

1. **Text-model scores:** XLM-R-base log-softmax scores and NB-SVM scores, each divided by its overall standard deviation on the scored set.
2. **ID-neighbour labels:** label shares among the 1, 2, 5, 10, 25, 50 and 100 nearest *other* training rows by ID; a distance-weighted share over the nearest 10; log ID distance to the two nearest.
3. **Neighbour text scores:** mean NB-SVM scores of the 2, 10 and 50 nearest rows by ID among all train and test rows. Train-row scores are out-of-fold (stage 3, five grouped folds); test-row scores come from the full-data NB-SVM. This lets unlabelled neighbours contribute.
4. **Post signals:** contains Bangla script, contains Latin script, log character length, and the Bangla share among the 10 nearest rows.

Leakage controls:

- A row's own label is never used for its own features.
- Text-model scores for stacking rows are always out-of-sample.
- The neighbour pool for training rows is "all other training rows", the same density test rows see.

## Stacker

- Training rows: the 19,127 rows with out-of-sample XLM-R logits (inner validation plus outer holdout).
- Models: multinomial logistic regression (`C=1`) and a histogram gradient-boosting classifier (learning rate 0.03, 15 leaves, 300 iterations, L2 1.0, minimum 50 samples per leaf). Their log-probabilities are averaged.
- Class offsets `[0, o1, o2]` are added to the averaged log-probabilities to maximise Macro F1. They are tuned by 3-fold cross-validation on the stacker's training rows over a grid from −1 to 1 in steps of 0.1. The final offsets are `[0, 0.2, 0.5]`.

## Results

| Model | Outer holdout | Kaggle public |
| --- | ---: | ---: |
| TF-IDF + LinearSVC (submission 1) | 0.5114 | 0.50248 |
| NB-SVM views (submission 4) | 0.5377 | 0.54759 |
| XLM-R-base (submission 5) | 0.5590 | 0.56654 |
| XLM-R-base + NB-SVM blend (submission 7) | 0.5798 | 0.59162 |
| + ID-neighbour labels, LR stack (submission 8) | 0.6803 | 0.69770 |
| **Submission 9 (this package)** | **0.6919** | **0.70190** |

Submission 9, outer holdout per-class F1: explicit 0.778, subtle 0.673, neutral 0.625.

Robustness check: repeated nested 5-fold CV over the 19,127 stacking rows, with offsets re-tuned inside each training fold, gives 0.6895, 0.6918 and 0.6897 (mean 0.6903, std 0.0010). The small spread and the agreement with the outer-holdout and public scores indicate the stacker is not overfitting its tuning data.

## Limitations

- The ID-neighbour gain depends on the test set sharing the ID ordering of the training set. That holds across the whole test ID range, so it should apply equally to the private leaderboard, but it reflects how the dataset was assembled rather than the text alone. The text-only pipeline (submission 7) is documented as the content-only baseline.
- The outer holdout informed several earlier experiments, so it is a development estimate. The nested CV provides a second estimate.
- XLM-R training on GPU is not bit-for-bit deterministic; the package includes the recorded logits so the submitted file can be reproduced exactly on CPU.
