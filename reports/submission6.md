# Submission 6: XLM-R-large + NB-SVM blend

Status: ready to launch on Kaggle GPU. No results yet.

## Why this model

Submission 5 showed the transformer is the strongest single model (outer 0.55903 vs 0.53766 for the NB-SVM views). Two cheap, well-understood gains remain:

1. **Larger backbone.** XLM-R-large has the same tokenizer and multilingual coverage (Bangla, English, romanized Banglish) as base, with roughly 3× the capacity. Texts are short (median 8 words, 99th percentile 58), so dynamic padding keeps T4 cost manageable.
2. **Blend with the submission 4 NB-SVM views.** Character n-gram models capture spelling variants in Banglish and code-mixed text that subword transformers often miss. Their errors differ from the transformer's, and they cost about one CPU minute per phase.

Class offsets are tuned together with the blend weight because Macro F1 rewards recall on the minority neutral class (12% of rows), where submission 5 was weakest (F1 0.490).

## Method

`src/train_submission6.py`, standalone, same splits as submissions 1–5.

- Transformer: `FacebookAI/xlm-roberta-large` at revision `c23d21b0620b635a76227c604d44e43a9f0ee389`. Learning rate 1e-5, batch 16 × accumulation 2 (effective 32), 3 epochs, 10% warmup + linear decay, max length 128, seed 42, square-root inverse-frequency class-weighted loss (identical to submission 5). CUDA mixed float16.
- NB-SVM: the three submission 4 views (word 1–3 + char_wb 2–6; word 1–2 + char 2–5; char_wb 2–6), `C=0.1`, `alpha=1.0`, equal weights, scores scaled by fitting-row std. Runs in a parallel CPU process.
- Blend: `w · z(log_softmax(transformer)) + (1−w) · z(NB-SVM) + [0, o1, o2]`, where `z` divides by the scored set's overall standard deviation (label-free). Grid: `w ∈ {0, 0.05, …, 1}`, `o1, o2 ∈ {−0.3, −0.25, …, 0.3}` (3,549 settings), chosen on inner validation only.
- Protocol: inner phase selects epoch count and blend settings → fresh outer-fit model is scored once on the outer holdout → CSV only if the blended outer Macro F1 exceeds submission 5's 0.5590271875 → final transformer and NB-SVM refit on all 47,817 rows.
- Safety: training stops after the inner phase if the large model collapses (inner Macro F1 ≤ 0.40). Existing outputs are never overwritten; reload is verified on test predictions.

The report records the outer Macro F1 of XLM-R-large alone, NB-SVM alone, and the blend with and without offsets, so the source of any gain is visible.

## Run

```bash
kaggle kernels push -p kaggle/submission6-script
kaggle kernels status swarlok/submission-6-xlm-r-large-nb-svm
kaggle kernels output swarlok/submission-6-xlm-r-large-nb-svm -p models/submission6_kaggle_results
```

Expected T4 runtime is about 2.5–3.5 hours (three transformer phases at roughly 3× submission 5's per-epoch time). If memory is short, use `--batch-size 8 --accumulation 4`.

Outputs: `submissions/submission6_xlmr_large_nbsvm.csv`, `reports/submission6_metrics.json`, `data/processed/submission6_validation.csv`, `models/submission6/`.

## Verification before launch

A CPU dry run on the real data with a tiny randomly initialised XLM-R (real large tokenizer, 1 epoch) completed every phase: inner selection, NB-SVM worker, blend search, outer gate, final refit, reload check and CSV write. The NB-SVM worker reproduced submission 4's inner Macro F1 (0.531 before offsets vs 0.532 recorded with offsets). This checks the pipeline, not model quality.
