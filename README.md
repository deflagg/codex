# carlos: Tiny GPT Training Harness

`carlos.py` trains a compact decoder-only transformer (~100K parameters) on a single UTF-8 text corpus using byte-level tokens. The script covers dataset preparation, training, evaluation, and sampling while keeping dependencies minimal.

## Requirements

Install dependencies before running any commands:

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Workflow Overview

1. **Prepare** – split corpus, tokenize, and pack sequences.
2. **Train** – run AdamW with cosine schedule, periodic validation, checkpoints, and auto text samples.
3. **Eval** – compute loss/perplexity/bits-per-byte on any split.
4. **Sample** – generate text from a checkpoint.

All artifacts are stored under a working directory (default `./carlos_run/`).

## 1. Prepare Dataset

```bash
python .\carlos.py prepare --input corpus_plain.txt --workdir ./carlos_run

python carlos.py prepare \
  --input corpus_plain.txt \
  --workdir ./carlos_run \
  --seq_len 512 \
  --normalize_nfkc  # optional Unicode normalization
```

Outputs include:
- `data/train.npy`, `data/val.npy`, `data/test.npy`
- `splits.json` with byte offsets and metadata
- `tokenizer.json` defining the byte-level vocabulary
- `rng_state.json` capturing seeds

## 2. Train the Model

```bash
python carlos.py train \
  --workdir ./carlos_run \
  --precision auto \
  --micro_batch_size 4 \
  --grad_accum_steps 4 \
  --sample_prefix "\n" \
  --sample_tokens 200
```

During training:
- Logs append to `metrics.jsonl`
- Checkpoints write to `checkpoints/step_XXXXXX.pt`
- Best checkpoint mirrors to `checkpoints/best.pt`
- Sample generations saved in `samples/`

The baseline configuration matches the provided `CarlosConfig` (d_model=64, n_layers=2, n_heads=4, d_ff=192).

### Optional: Weights & Biases logging

If a W&B project is supplied, training metrics and samples stream to Weights & Biases. Environment variables from `.env` are loaded automatically, so setting `WANDB_API_KEY` and `WANDB_PROJECT` there is enough. Override or enable logging per run with CLI flags:

```bash
python carlos.py train --workdir ./carlos_run

python carlos.py train \
  --workdir ./carlos_run \
  --wandb_project tiny-models \
  --wandb_run_name carlos-demo \
  --wandb_tags baseline byte-level
```

Use `--wandb_mode offline` for local logging or `--wandb_mode disabled` to skip even if the environment defines a project. When neither CLI nor environment specifies a project, W&B stays inactive.

## 3. Evaluate a Checkpoint

```bash
 python carlos.py eval --workdir ./carlos_run --checkpoint best --split test
 
python carlos.py eval \
  --workdir ./carlos_run \
  --checkpoint best \
  --split test
```

Outputs loss, perplexity, and bits-per-byte for the selected split. Supply a specific `.pt` path to inspect other checkpoints.

## 4. Generate Text

```bash
python carlos.py sample --prompt INGREDIENTS

python carlos.py sample \
  --workdir ./carlos_run \
  --checkpoint best \
  --prompt "\n" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9 \
  --top_k 50
```

If no prompt is provided, the script defaults to BOS (when available) or a newline.

## Tips

- Use the `--add_bos` / `--add_eos` flags during `prepare` if you need explicit boundary tokens.
- Adjust throughput via `--micro_batch_size`, `--grad_accum_steps`, or `--seq_len` (256 works on tighter memory).
- The scheduler uses 1–2k warmup steps and cosine decay down to 1e-5 by default; tune with `--warmup_steps`, `--peak_lr`, and `--min_lr`.
- `metrics.jsonl` is newline-delimited JSON ready for plotting or analysis.

Happy training!



