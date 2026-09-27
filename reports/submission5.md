# Submission 5: XLM-R experiment

Local training started on 2026-09-27 at 21:34 Bangladesh time (15:34 UTC), then was stopped at the requested switch to Kaggle GPU. The local process group was terminated; its files are preserved. No validation or public score is available. Submission 4 remains the best completed submission, with validation Macro F1 0.5376578865 and public Macro F1 0.54759.

Kaggle accepted version 2 of the private Python script job: [Submission 5 XLM-R](https://www.kaggle.com/code/jahidul183019/submission-5-xlm-r). It is queued for a GPU worker. Downloaded server metadata confirms `kernel_type: script`, private visibility, GPU enabled, internet enabled, the T4 machine type and the attached competition data. Validation and submission results are pending.

Phone and identity verification are complete, and the Kaggle CLI is authenticated. The local `.ipynb` draft was removed. The launch configuration is `kaggle/submission5-script/kernel-metadata.json`; no notebook is used for this training job.

## Training

`src/train_submission5.py` fine-tunes `FacebookAI/xlm-roberta-base` at revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`. It uses only the provided competition CSVs for downstream training. Text retains its original case, punctuation and script; Unicode and whitespace are normalized. Split grouping uses the same lowercased normalization as submissions 1–4.

- Maximum sequence length: 128 tokens.
- Batch size: 8; gradient accumulation: 4 (effective batch size 32).
- AdamW: learning rate 0.00002, weight decay 0.01, 10% warmup followed by linear decay.
- Loss: cross entropy with square-root inverse-frequency class weights, calculated only from each phase's training labels.
- Seed: 42. Three inner-training epochs; the highest inner Macro F1 selects the epoch count.

The accumulated loss is divided by the number of examples in each optimizer window. It is therefore a fixed per-example weighted mean rather than PyTorch's weighted-mean reduction, which divides by the sum of observed class weights. The class weights are mean-normalized to 1; this choice keeps partial accumulation windows consistent and is recorded for comparisons with other training implementations.

A fresh model then trains on the outer fitting rows for that epoch count. The outer holdout matches submission 4 exactly. The learning-rate schedule spans each phase's full duration. If outer Macro F1 exceeds submission 4, another fresh model trains on all competition training rows and generates `submissions/submission5_xlmr.csv`. Otherwise, the script saves evaluation results without generating a submission. This threshold is a development decision on a reused holdout, not proof of better private performance. Additional grouped-fold confirmation remains outstanding.

Each epoch retains its configuration, tokenizer, metrics and validation logits. After a phase finishes, the selected epoch keeps its full model weights and redundant weights from the other epochs are removed. `checkpoint_retention.json` records the retained checkpoint and removed files. The pinned initial model, copied source, settings, split indices and history are sufficient to reproduce the discarded weights while keeping Kaggle output size manageable. Existing run directories are rejected. Final predictions are checked after reloading the saved model and the CSV is checked against sample-submission IDs and ordering. Kaggle uploads remain manual.

## Runtime and execution

A six-step benchmark on the Mac's MPS GPU measured approximately 1.05 seconds per batch after warmup at length 128. This suggests roughly 6–13 hours for the complete workflow, depending on the selected epoch count. Actual dynamic padding and gradient accumulation may change throughput. The benchmark is not a validation run and its losses are not performance evidence.

The sandbox cannot access MPS; the benchmark succeeded outside the sandbox. The local machine has 16 GB of unified memory. Initial checkpoint files are in `models/submission5_benchmark/pretrained`; the benchmark saves them before its diagnostic training steps.

Local background launch:

```bash
python3 src/run_submission5_local.py --run-name submission5_local
```

Progress is written to `models/<run-name>_launch/training.log`, with process information in `launch.json` and exit status in `completion.json` when the worker ends. A local run named `submission5_local` started at 17:34 UTC on 2026-09-27 while the Kaggle job remained queued. Keep the Mac connected to power with its lid open. The launcher prevents idle system sleep while training. No OpenAI API is used by the training process.

For a foreground run with a new name:

```bash
python3 src/train_submission5.py --run-name submission5_repeat --device mps
```

The script is standalone and can also run on a Kaggle GPU with the competition data attached privately. CUDA uses mixed float16 precision with gradient scaling. The script automatically finds the attached competition directory and validates all three input hashes.

```bash
kaggle kernels push -p kaggle/submission5-script
```

The script installs its pinned training libraries in Kaggle before importing them and retains Kaggle's CUDA-enabled PyTorch. It records the actual PyTorch and other package versions in the run manifest. The pinned pretrained checkpoint is downloaded from Hugging Face.

Check the remote run and download its outputs after completion:

```bash
kaggle kernels status jahidul183019/submission-5-xlm-r
kaggle kernels output jahidul183019/submission-5-xlm-r -p models/submission5_kaggle_results
```

Pushing again creates another remote run. Check the current run before relaunching to avoid consuming GPU quota unnecessarily.

The stage option does not persist `/kaggle/working` across separate Kaggle sessions. To resume with `--stage outer` or `--stage final` in a new session, first save and reattach the preceding session's outputs so the complete run directory, metrics and validation predictions are available at the same working paths. The default `--stage all` completes all phases in one Kaggle run.

## Checks completed before training

- Raw input hashes match submission 4.
- Outer validation IDs and original labels match submission 4 exactly.
- Split sizes: 28,690 inner training; 9,564 inner validation; 38,254 outer training; 9,563 outer validation.
- Small CPU integration tests cover grouped-duplicate separation, a partial gradient-accumulation window, selected-weight retention, validation-logit retention, model reload predictions and refusal to overwrite a phase.

These checks verify the pipeline, not whether XLM-R improves competition performance.
