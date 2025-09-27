#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carlos: compact GPT-style decoder-only LM harness.

This script prepares byte-level datasets from a single UTF-8 file, trains a
small Pre-LN transformer with rotary position embeddings, evaluates checkpoints,
and generates samples. Dependencies: PyTorch and NumPy.
"""

import argparse
import json
import math
import os
import random
import time
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None

if load_dotenv is not None:
    load_dotenv()

HAS_WANDB = wandb is not None


@dataclass
class CarlosConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 192
    dropout: float = 0.1
    seq_len: int = 512
    use_bos: bool = False
    use_eos: bool = False
    peak_lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 1500
    max_steps: int = 10000
    micro_batch_size: int = 4
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    early_stop_patience: int = 5
    max_epochs: Optional[int] = None
    device: Optional[str] = None
    precision: str = "auto"  # auto, fp16, bf16, fp32
    seed: int = 1234
    label_smoothing: float = 0.0


class ByteTokenizer:
    """Simple byte-level tokenizer with optional BOS/EOS."""

    def __init__(self, add_bos: bool = False, add_eos: bool = False):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.byte_to_id = {i: i for i in range(256)}
        idx = 256
        if add_bos:
            self.bos_id = idx
            idx += 1
        else:
            self.bos_id = None
        if add_eos:
            self.eos_id = idx
            idx += 1
        else:
            self.eos_id = None
        self.vocab_size = idx

    def encode_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8).astype(np.int64)

    def decode(self, tokens: Iterable[int]) -> str:
        byte_vals = [t for t in tokens if 0 <= t <= 255]
        return bytes(byte_vals).decode("utf-8", errors="ignore")

    def to_json(self) -> Dict:
        return {
            "type": "byte",
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "vocab_size": self.vocab_size,
            "byte_to_id": {str(k): v for k, v in self.byte_to_id.items()},
        }

    @classmethod
    def from_json(cls, payload: Dict) -> "ByteTokenizer":
        tok = cls(add_bos=payload.get("add_bos", False), add_eos=payload.get("add_eos", False))
        return tok

    def bos_token_id(self) -> Optional[int]:
        return self.bos_id

    def eos_token_id(self) -> Optional[int]:
        return self.eos_id

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(path: str, normalize_nfkc: bool = False) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalize_nfkc:
        text = unicodedata.normalize("NFKC", text)
    return text


def compute_split_offsets(total_bytes: int) -> Dict[str, Tuple[int, int]]:
    if total_bytes < 3:
        raise ValueError("Corpus must contain at least 3 bytes for train/val/test splits")
    train_end = int(total_bytes * 0.90)
    train_end = max(1, min(train_end, total_bytes - 2))
    val_size = int(total_bytes * 0.05)
    val_size = max(1, val_size)
    val_end = train_end + val_size
    val_end = max(train_end + 1, min(val_end, total_bytes - 1))
    splits = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_bytes),
    }
    return splits


class PackedDataset(Dataset):
    """Dataset that packs contiguous token streams into fixed-length blocks."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        if tokens.ndim != 1:
            tokens = tokens.reshape(-1)
        self.seq_len = seq_len
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        usable = self.tokens.numel() - 1
        self.num_sequences = usable // seq_len
        if self.num_sequences <= 0:
            raise ValueError("Not enough tokens to form a single sequence with seq_len={}".format(seq_len))

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x.clone(), y.clone()


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class CarlosSelfAttention(nn.Module):
    def __init__(self, config: CarlosConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.rotary = RotaryEmbedding(self.head_dim)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rotary(T, x.device, dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        return out


class CarlosMLP(nn.Module):
    def __init__(self, config: CarlosConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: CarlosConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CarlosSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = CarlosMLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, dtype)
        x = x + self.mlp(self.ln2(x))
        return x


class CarlosModel(nn.Module):
    def __init__(self, config: CarlosConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.config.seq_len:
            raise ValueError(f"Sequence length {T} exceeds configured maximum {self.config.seq_len}")
        x = self.token_emb(idx)
        mask = self._causal_mask(T, x.device)
        dtype = x.dtype
        for block in self.blocks:
            x = block(x, mask, dtype)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        idx = idx.to(device)
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.seq_len :]
            logits, _ = self(context)
            next_token = sample_next_token(logits[:, -1, :], temperature, top_p, top_k)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def cosine_lr(step: int, config: CarlosConfig) -> float:
    if step < config.warmup_steps:
        return config.peak_lr * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.peak_lr - config.min_lr) * cosine


def load_tokens(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def create_dataloader(tokens_path: str, seq_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    tokens = load_tokens(tokens_path)
    dataset = PackedDataset(tokens, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def evaluate_model(
    model: CarlosModel,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    avg = float(np.mean(losses))
    return {
        "loss": avg,
        "ppl": math.exp(avg),
        "bpb": avg / math.log(2),
    }


def load_config(path: str) -> CarlosConfig:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return CarlosConfig(**payload)


def save_config(path: str, config: CarlosConfig) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)


def save_checkpoint(
    path: str,
    model: CarlosModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    config: CarlosConfig,
    best_val_loss: float,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(config),
        "best_val_loss": best_val_loss,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: CarlosModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Optional[str] = None,
) -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def maybe_autocast_dtype(device: torch.device, precision: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unknown precision option: {precision}")


def init_wandb_run(
    args: argparse.Namespace,
    config: CarlosConfig,
    tokens_per_step: int,
    param_count: int,
    dataset_meta: Dict[str, Any],
):
    mode = getattr(args, "wandb_mode", None)
    if mode == "disabled":
        return None
    project = getattr(args, "wandb_project", None) or os.getenv("WANDB_PROJECT")
    if not project:
        return None
    if not HAS_WANDB:
        raise ImportError(
            "Weights & Biases logging requested, but the wandb package is not installed. "
            "Install wandb or disable logging with --wandb_mode disabled."
        )
    resolved_mode = mode or os.getenv("WANDB_MODE")
    if resolved_mode == "disabled":
        return None
    settings = None
    if resolved_mode:
        settings = wandb.Settings(mode=resolved_mode)
    run_config = asdict(config).copy()
    run_config.update(
        {
            "tokens_per_step": tokens_per_step,
            "param_count": param_count,
            "dataset_meta": dataset_meta,
        }
    )
    run = wandb.init(
        project=project,
        name=getattr(args, "wandb_run_name", None),
        dir=getattr(args, "workdir", None),
        tags=getattr(args, "wandb_tags", None),
        config=run_config,
        settings=settings,
    )
    return run


def wandb_log(run, payload: Dict[str, Any], step: int) -> None:
    if run is not None:
        run.log(payload, step=step)

def cmd_prepare(args: argparse.Namespace) -> None:
    ensure_dir(args.workdir)
    data_dir = os.path.join(args.workdir, "data")
    ensure_dir(data_dir)
    tokenizer = ByteTokenizer(add_bos=args.add_bos, add_eos=args.add_eos)
    set_seed(args.seed)
    text = load_corpus(args.input, normalize_nfkc=args.normalize_nfkc)
    corpus_bytes = text.encode("utf-8")
    total_bytes = len(corpus_bytes)
    if total_bytes <= args.seq_len:
        raise ValueError("Corpus is too small relative to seq_len; need more bytes")
    splits = compute_split_offsets(total_bytes)
    split_meta = {}
    for name, (start, end) in splits.items():
        chunk = corpus_bytes[start:end]
        tokens = tokenizer.encode_bytes(chunk).astype(np.uint16)
        out_path = os.path.join(data_dir, f"{name}.npy")
        np.save(out_path, tokens, allow_pickle=False)
        token_count = int(tokens.shape[0])
        windows = max(0, (token_count - 1) // args.seq_len)
        split_meta[name] = {
            "start": int(start),
            "end": int(end),
            "bytes": int(end - start),
            "tokens": token_count,
            "windows": windows,
            "npy": os.path.relpath(out_path, args.workdir),
        }
    meta = {
        "prepared_at": time.time(),
        "input_path": os.path.abspath(args.input),
        "total_bytes": total_bytes,
        "seq_len": args.seq_len,
        "normalize_nfkc": args.normalize_nfkc,
        "splits": split_meta,
        "seed": args.seed,
    }
    with open(os.path.join(args.workdir, "splits.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(args.workdir, "tokenizer.json"), "w", encoding="utf-8") as fh:
        json.dump(tokenizer.to_json(), fh, indent=2)
    rng_state = {"prepare_seed": args.seed, "saved_at": time.time()}
    with open(os.path.join(args.workdir, "rng_state.json"), "w", encoding="utf-8") as fh:
        json.dump(rng_state, fh, indent=2)
    print(
        f"Prepared dataset -> train windows: {split_meta['train']['windows']}, val windows: {split_meta['val']['windows']}, "
        f"test windows: {split_meta['test']['windows']}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    splits_path = os.path.join(args.workdir, "splits.json")
    tokenizer_path = os.path.join(args.workdir, "tokenizer.json")
    if not os.path.exists(splits_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Run prepare first to create splits.json and tokenizer.json")
    with open(splits_path, "r", encoding="utf-8") as fh:
        split_meta = json.load(fh)
    with open(tokenizer_path, "r", encoding="utf-8") as fh:
        tokenizer = ByteTokenizer.from_json(json.load(fh))

    config_path = os.path.join(args.workdir, "config.json")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = CarlosConfig()

    config.vocab_size = tokenizer.vocab_size
    config.use_bos = tokenizer.add_bos
    config.use_eos = tokenizer.add_eos
    config.seq_len = split_meta.get("seq_len", config.seq_len)
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.precision is not None:
        config.precision = args.precision
    if args.seed is not None:
        config.seed = args.seed
    if args.peak_lr is not None:
        config.peak_lr = args.peak_lr
    if args.min_lr is not None:
        config.min_lr = args.min_lr

    save_config(config_path, config)

    set_seed(config.seed)

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = maybe_autocast_dtype(device, config.precision)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    train_path = os.path.join(args.workdir, split_meta["splits"]["train"]["npy"])
    val_path = os.path.join(args.workdir, split_meta["splits"]["val"]["npy"])

    train_loader = create_dataloader(train_path, config.seq_len, config.micro_batch_size, shuffle=True)
    val_loader = create_dataloader(val_path, config.seq_len, config.micro_batch_size, shuffle=False)

    model = CarlosModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    tokens_per_step = config.micro_batch_size * config.grad_accum_steps * config.seq_len
    print(f"Effective tokens/step: {tokens_per_step}")

    raw_splits = split_meta.get("splits", {})
    dataset_meta = {
        name: {k: v for k, v in info.items() if k in {"tokens", "windows", "bytes"}}
        for name, info in raw_splits.items()
    }
    wandb_run = init_wandb_run(args, config, tokens_per_step, param_count, dataset_meta)
    if wandb_run is not None:
        wandb.watch(model, log="gradients", log_freq=config.log_interval)
        for split_name, info in dataset_meta.items():
            wandb_run.config.update({f"data/{split_name}": info})

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    metrics_path = os.path.join(args.workdir, "metrics.jsonl")
    checkpoints_dir = os.path.join(args.workdir, "checkpoints")
    samples_dir = os.path.join(args.workdir, "samples")
    ensure_dir(checkpoints_dir)
    ensure_dir(samples_dir)

    best_val_loss: Optional[float] = None
    patience = 0
    running_loss = 0.0
    start_time = time.time()
    train_iter = iter(train_loader)
    steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps) if len(train_loader) else 1

    try:
        for step in range(config.max_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(config.grad_accum_steps):
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    xb, yb = next(train_iter)
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                    _, loss = model(xb, yb)
                    loss = loss / config.grad_accum_steps
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += loss.item() * config.grad_accum_steps
            if use_scaler:
                scaler.unscale_(optimizer)
            if config.grad_clip is not None and config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            lr = cosine_lr(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            running_loss += step_loss
            global_step = step + 1
            current_epoch = (global_step - 1) / steps_per_epoch if steps_per_epoch else 0.0
            if wandb_run is not None:
                batch_loss = step_loss / max(1, config.grad_accum_steps)
                wandb_log(
                    wandb_run,
                    {
                        "train/batch_loss": batch_loss,
                        "train/batch": global_step,
                        "train/lr": lr,
                        "train/epoch": current_epoch,
                    },
                    step=global_step,
                )

            if global_step % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                elapsed = time.time() - start_time
                metrics = {
                    "step": global_step,
                    "split": "train",
                    "loss": avg_loss,
                    "ppl": math.exp(avg_loss),
                    "bpb": avg_loss / math.log(2),
                    "lr": lr,
                    "tokens_per_step": tokens_per_step,
                    "elapsed_sec": elapsed,
                }
                with open(metrics_path, "a", encoding="utf-8") as log_fh:
                    log_fh.write(json.dumps(metrics) + "\n")
                print(
                    f"step {global_step:6d} | train loss {avg_loss:.4f} | ppl {metrics['ppl']:.2f} | bpb {metrics['bpb']:.3f} | lr {lr:.6f}"
                )
                if wandb_run is not None:
                    wandb_log(
                        wandb_run,
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/epoch": current_epoch,
                        },
                        step=global_step,
                    )
                running_loss = 0.0
            if global_step % config.eval_interval == 0 or global_step == config.max_steps:
                val_stats = evaluate_model(model, val_loader, device, amp_dtype)
                val_metrics = {"step": global_step, "split": "val", **val_stats}
                with open(metrics_path, "a", encoding="utf-8") as log_fh:
                    log_fh.write(json.dumps(val_metrics) + "\n")
                print(
                    f"step {global_step:6d} | val loss {val_stats['loss']:.4f} | ppl {val_stats['ppl']:.2f} | bpb {val_stats['bpb']:.3f}"
                )
                if wandb_run is not None:
                    wandb_log(
                        wandb_run,
                        {
                            "val/loss": val_stats["loss"],
                            "val/epoch": current_epoch,
                        },
                        step=global_step,
                    )

                checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:06d}.pt")
                save_checkpoint(checkpoint_path, model, optimizer, scaler if use_scaler else None, global_step, config, val_stats["loss"])

                improved = best_val_loss is None or val_stats["loss"] < best_val_loss
                if improved:
                    best_val_loss = val_stats["loss"]
                    patience = 0
                    best_path = os.path.join(checkpoints_dir, "best.pt")
                    save_checkpoint(best_path, model, optimizer, scaler if use_scaler else None, global_step, config, best_val_loss)
                    sample_prefix = args.sample_prefix.encode("utf-8") if args.sample_prefix else b"\n"
                    prompt_tokens = torch.tensor(tokenizer.encode_bytes(sample_prefix), dtype=torch.long).unsqueeze(0)
                    generated = model.generate(
                        prompt_tokens,
                        max_new_tokens=args.sample_tokens,
                        temperature=args.sample_temperature,
                        top_p=args.sample_top_p,
                        top_k=args.sample_top_k,
                    )
                    text = tokenizer.decode(generated[0].tolist())
                    sample_path = os.path.join(samples_dir, f"step_{global_step:06d}.txt")
                    with open(sample_path, "w", encoding="utf-8") as fh:
                        fh.write(text)
                    if wandb_run is not None:
                        wandb_log(wandb_run, {"val/best_loss": best_val_loss}, step=global_step)
                        wandb_log(wandb_run, {"samples/text": text}, step=global_step)
                else:
                    patience += 1
                    if patience >= config.early_stop_patience:
                        print("Early stopping triggered.")
                        break

            if config.max_epochs is not None:
                steps_per_epoch = len(train_loader) // config.grad_accum_steps
                if steps_per_epoch > 0 and global_step >= steps_per_epoch * config.max_epochs:
                    print("Max epochs reached.")
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    print("Training finished.")


def cmd_eval(args: argparse.Namespace) -> None:
    config_path = os.path.join(args.workdir, "config.json")
    tokenizer_path = os.path.join(args.workdir, "tokenizer.json")
    splits_path = os.path.join(args.workdir, "splits.json")
    checkpoint_path = args.checkpoint
    if checkpoint_path == "best":
        checkpoint_path = os.path.join(args.workdir, "checkpoints", "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(config_path)
    with open(tokenizer_path, "r", encoding="utf-8") as fh:
        tokenizer = ByteTokenizer.from_json(json.load(fh))
    with open(splits_path, "r", encoding="utf-8") as fh:
        split_meta = json.load(fh)

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = maybe_autocast_dtype(device, config.precision)

    model = CarlosModel(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    data_rel = split_meta["splits"][args.split]["npy"]
    data_path = os.path.join(args.workdir, data_rel)
    loader = create_dataloader(data_path, config.seq_len, config.micro_batch_size, shuffle=False)

    stats = evaluate_model(model, loader, device, amp_dtype)
    print(
        f"{args.split} loss {stats['loss']:.4f} | ppl {stats['ppl']:.2f} | bpb {stats['bpb']:.3f}"
    )


def cmd_sample(args: argparse.Namespace) -> None:
    config_path = os.path.join(args.workdir, "config.json")
    tokenizer_path = os.path.join(args.workdir, "tokenizer.json")
    checkpoint_path = args.checkpoint
    if checkpoint_path == "best":
        checkpoint_path = os.path.join(args.workdir, "checkpoints", "best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(config_path)
    with open(tokenizer_path, "r", encoding="utf-8") as fh:
        tokenizer = ByteTokenizer.from_json(json.load(fh))

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarlosModel(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    if args.prompt:
        prompt = args.prompt.encode("utf-8")
    elif tokenizer.bos_token_id() is not None:
        prompt = bytes([tokenizer.bos_token_id()])
    else:
        prompt = b"\n"
    prompt_tokens = torch.tensor(tokenizer.encode_bytes(prompt), dtype=torch.long).unsqueeze(0)
    generated = model.generate(
        prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="carlos: tiny GPT training harness")
    sub = parser.add_subparsers(dest="command")

    prep = sub.add_parser("prepare", help="Prepare dataset from corpus")
    prep.add_argument("--input", required=True, help="Path to UTF-8 text corpus")
    prep.add_argument("--workdir", default="./carlos_run", help="Working directory")
    prep.add_argument("--seq_len", type=int, default=512, help="Sequence length for packing")
    prep.add_argument("--normalize_nfkc", action="store_true", help="Apply Unicode NFKC normalization")
    prep.add_argument("--add_bos", action="store_true", help="Add BOS token to vocabulary")
    prep.add_argument("--add_eos", action="store_true", help="Add EOS token to vocabulary")
    prep.add_argument("--seed", type=int, default=1234, help="Seed used during preparation metadata")
    prep.set_defaults(func=cmd_prepare)

    train = sub.add_parser("train", help="Train the carlos model")
    train.add_argument("--workdir", default="./carlos_run", help="Working directory")
    train.add_argument("--max_steps", type=int, default=None, help="Training steps")
    train.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    train.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    train.add_argument("--micro_batch_size", type=int, default=None, help="Per-device micro batch size")
    train.add_argument("--eval_interval", type=int, default=250, help="Validation interval in steps")
    train.add_argument("--log_interval", type=int, default=None, help="Logging interval in steps")
    train.add_argument("--precision", type=str, default=None, help="Numerical precision: auto, fp16, bf16, fp32")
    train.add_argument("--seed", type=int, default=None, help="Training seed")
    train.add_argument("--peak_lr", type=float, default=None, help="Peak learning rate")
    train.add_argument("--min_lr", type=float, default=None, help="Min learning rate")
    train.add_argument("--sample_prefix", type=str, default="\n", help="Prompt prefix for samples")
    train.add_argument("--sample_tokens", type=int, default=200, help="Sample length in new tokens")
    train.add_argument("--sample_temperature", type=float, default=0.8, help="Sampling temperature")
    train.add_argument("--sample_top_p", type=float, default=0.9, help="Top-p for sampling")
    train.add_argument("--sample_top_k", type=int, default=50, help="Top-k for sampling")
    train.add_argument("--wandb_project", type=str, default='carlos', help="Weights & Biases project name")
    train.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    train.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B mode override")
    train.add_argument("--wandb_tags", nargs="*", default=None, help="Tags to attach to the W&B run")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="Evaluate a checkpoint")
    ev.add_argument("--workdir", default="./carlos_run", help="Working directory")
    ev.add_argument("--checkpoint", default="best", help="Checkpoint path or 'best'")
    ev.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to evaluate")
    ev.set_defaults(func=cmd_eval)

    sample = sub.add_parser("sample", help="Generate text from a checkpoint")
    sample.add_argument("--workdir", default="./carlos_run", help="Working directory")
    sample.add_argument("--checkpoint", default="best", help="Checkpoint path or 'best'")
    sample.add_argument("--prompt", type=str, default=None, help="Prompt text")
    sample.add_argument("--max_new_tokens", type=int, default=200, help="Tokens to generate")
    sample.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    sample.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    sample.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    sample.set_defaults(func=cmd_sample)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()


























