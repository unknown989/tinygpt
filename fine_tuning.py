"""
tiny_gpt_finetune.py
────────────────────
Fine-tunes a TinyGPT checkpoint on a Q&A CSV dataset.

Expected CSV columns (at minimum):
    - prompt_name  : the question  (e.g. "Does the electoral college work?")
    - text         : the answer    (e.g. "The Electoral College is a complex system...")

Optional columns (ignored during training but kept for reference):
    - label, source, RDizzI3_se...

Usage
─────
1. Set PRETRAINED_CHECKPOINT to the path of your epoch-9 .pth file.
2. Set CSV_PATH to your dataset .csv file.
3. Optionally adjust QUESTION_COL / ANSWER_COL if your column names differ.
4. Run: python tiny_gpt_finetune.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import csv

# ── Precision settings ─────────────────────────────────────────────────────────
torch.backends.cudnn.conv.fp32_precision  = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.benchmark            = True

enc = tiktoken.get_encoding("gpt2")

# ── Config ─────────────────────────────────────────────────────────────────────
PRETRAINED_CHECKPOINT = "checkpoints/tiny_gpt_epoch_9.pth"  # path to your base checkpoint
CSV_PATH              = "/kaggle/input/datasets/etiennekaiser/gemini-pro-llm-daigt-dataset/train_essays_v1.csv"           # <-- SET YOUR CSV PATH HERE (or leave blank to be prompted)
QUESTION_COL          = "prompt_name"
ANSWER_COL            = "text"

# Fine-tune hyperparameters (lower LR than pretraining — we don't want to "forget")
block_size       = 128
BATCH_SIZE       = 64
GRAD_ACCUM_STEPS = 2
NUM_WORKERS      = 4
LEARNING_RATE    = 5e-5   # ~6× lower than pretraining LR
NUM_EPOCHS       = 50
WARMUP_STEPS     = 100

# Q&A prompt template — this is what the model learns to complete
# Format:  Q: {question}\nA: {answer}<|endoftext|>
QA_TEMPLATE = "Q: {question}\nA: {answer}<|endoftext|>"

EOT_TOKEN = enc.encode("<|endoftext|>",allowed_special={'<|endoftext|>'})[0]   # token id 50256 in GPT-2 vocab


# ── Model (identical architecture to pretraining) ──────────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_head   = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj     = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = math.sqrt(self.head_dim)
            att   = (q @ k.transpose(-2, -1)) / scale
            att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att   = F.softmax(att, dim=-1)
            att   = self.drop(att)
            out   = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T   = idx.shape
        tok    = self.tok_emb(idx)
        pos    = self.pos_emb(torch.arange(T, device=idx.device))
        x      = self.drop(tok + pos)
        x      = self.blocks(x)
        x      = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ── Q&A Dataset ────────────────────────────────────────────────────────────────
class QADataset(Dataset):
    """
    Reads a CSV file and converts each row into a tokenised Q&A sequence.

    Training strategy — answer-only loss masking
    ─────────────────────────────────────────────
    We only compute loss on the *answer* tokens, not the question tokens.
    This prevents the model wasting capacity learning to reproduce questions
    and focuses it entirely on generating good answers.

    Layout of a single sample (token ids):
        [Q: <question tokens> \n A: <answer tokens> <|endoftext|>]
         ←── masked (label = -100) ───→ ←──── loss computed here ────→
    """

    def __init__(self, csv_path, enc, block_size,
                 question_col="prompt_name", answer_col="text"):
        self.enc        = enc
        self.block_size = block_size
        self.samples    = []  # list of (input_ids, label_ids) tensors

        # Token ids for the separator "\nA:" so we can find where answer starts
        separator_ids = enc.encode("\nA:",allowed_special={'<|endoftext|>'})

        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)

            missing = [c for c in [question_col, answer_col] if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"Column(s) not found in CSV: {missing}\n"
                    f"Available columns: {reader.fieldnames}"
                )

            skipped = 0
            for row in reader:
                question = row[question_col].strip()
                answer   = row[answer_col].strip()
                if not question or not answer:
                    skipped += 1
                    continue

                full_text = QA_TEMPLATE.format(question=question, answer=answer)
                tokens    = enc.encode(full_text,allowed_special={'<|endoftext|>'})

                # Truncate to block_size
                if len(tokens) > block_size:
                    tokens = tokens[:block_size]

                # Pad short sequences so DataLoader can batch them uniformly
                pad_len = block_size - len(tokens)
                tokens  = tokens + [EOT_TOKEN] * pad_len

                # ── Build labels (answer-only loss masking) ──────────────────
                # Find where "\nA:" separator starts inside the token sequence
                prefix_text = f"Q: {question}\nA:"
                prefix_ids  = enc.encode(prefix_text,allowed_special={'<|endoftext|>'})
                answer_start = len(prefix_ids)   # index of first answer token

                labels = [-100] * block_size      # -100 = ignored by cross_entropy
                for i in range(answer_start, block_size - 1):
                    labels[i] = tokens[i + 1]     # predict next token

                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                label_ids = torch.tensor(labels[:-1], dtype=torch.long)

                self.samples.append((input_ids, label_ids))

        if skipped:
            print(f"  Skipped {skipped} rows with empty question or answer.")
        print(f"  Loaded {len(self.samples)} Q&A pairs from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Helpers ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def answer(model, enc, question, device, max_new_tokens=150, temperature=0.7, top_k=40):
    """Generate an answer for a given question string."""
    model.eval()
    prompt = f"Q: {question}\nA:"
    idx    = torch.tensor(enc.encode(prompt,allowed_special={'<|endoftext|>'}), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / temperature
        v, _      = torch.topk(logits, top_k)
        logits[logits < v[:, -1, None]] = -float("Inf")
        probs    = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Stop at <|endoftext|>
        if idx_next.item() == EOT_TOKEN:
            break
        idx = torch.cat((idx, idx_next), dim=1)

    decoded = enc.decode(idx[0].tolist())
    # Return just the answer part
    if "\nA:" in decoded:
        return decoded.split("\nA:", 1)[1].strip()
    return decoded


def load_pretrained(path, model, device):
    """Load model weights from a pretraining checkpoint."""
    abs_path = os.path.abspath(path)
    print(f"\nLoading pretrained checkpoint: {abs_path}")

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {abs_path}\n"
            f"Set PRETRAINED_CHECKPOINT at the top of this script."
        )

    ckpt = torch.load(abs_path, map_location=device, weights_only=True)
    key  = 'model_state_dict' if 'model_state_dict' in ckpt else 'model'
    if key not in ckpt:
        raise KeyError(f"No model weights in checkpoint. Keys: {list(ckpt.keys())}")

    model.load_state_dict(ckpt[key])
    saved_epoch = ckpt.get('epoch', '?')
    saved_loss  = ckpt.get('loss',  '?')
    print(f"  Loaded weights from epoch {saved_epoch}, loss {saved_loss}")
    return ckpt


def resolve_csv_path(path):
    if path and os.path.isfile(path):
        return path
    if path:
        print(f"WARNING: CSV_PATH '{path}' not found.")
    print("\n--- CSV Dataset Selection ---")
    while True:
        p = input("Enter the full path to your Q&A .csv file: ").strip()
        if os.path.isfile(p):
            return p
        print(f"  Not found: '{p}'. Try again.")


def get_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    CUDA_CAPABILITY = torch.cuda.get_device_capability(0)
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA Capability: {CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]}")
else:
    CUDA_CAPABILITY = (0, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device  : {device}")

# ── Build model ────────────────────────────────────────────────────────────────
model = TinyGPT(
    vocab_size=50257,
    d_model=128,
    n_heads=8,
    n_layers=6,
    block_size=block_size,
    dropout=0.1,
)
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# torch.compile only for Ampere+ GPUs
if CUDA_CAPABILITY[0] >= 7 and hasattr(torch, 'compile'):
    model = torch.compile(model)
    print("torch.compile: enabled")
else:
    print(f"torch.compile: skipped (CUDA {CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]} < 7.0)")

# ── Load pretrained weights ────────────────────────────────────────────────────
load_pretrained(PRETRAINED_CHECKPOINT, model, device)

# ── Load CSV dataset ───────────────────────────────────────────────────────────
csv_path = resolve_csv_path(CSV_PATH)
print(f"\nBuilding Q&A dataset from: {csv_path}")

dataset = QADataset(
    csv_path=csv_path,
    enc=enc,
    block_size=block_size,
    question_col=QUESTION_COL,
    answer_col=ANSWER_COL,
)

if len(dataset) == 0:
    raise RuntimeError("Dataset is empty — check your CSV file and column names.")

loader = DataLoader(
    dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
    persistent_workers = True,
)

print(f"Q&A pairs      : {len(dataset):,}")
print(f"Batches/epoch  : {len(loader):,}")

# ── Optimizer & scheduler ──────────────────────────────────────────────────────
optimizer    = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps  = len(loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
scheduler    = get_warmup_scheduler(optimizer, WARMUP_STEPS, total_steps)
scaler       = torch.amp.GradScaler('cuda')

# ── Fine-tuning loop ───────────────────────────────────────────────────────────
os.makedirs("checkpoints_ft", exist_ok=True)
print(f"\nFine-tuning for {NUM_EPOCHS} epochs (LR={LEARNING_RATE}) ...")

# Quick sanity-check generation before training
print("\n[Pre-fine-tune sample]")
if len(dataset) > 0:
    sample_q = dataset.samples[0]  # just use the first question's tokens to reverse-engineer q
    print(answer(model, enc, "What is your question?", device, max_new_tokens=60))

global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss  = 0.0
    answer_loss = 0.0   # loss on answer tokens only (the meaningful metric)
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits, _ = model(x)       # (B, T, V)

            # Compute loss only on non-masked positions (y != -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100,     # skip question tokens
            )
            loss_acc = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss_acc).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item()

        if step % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            if torch.cuda.is_available():
                vram = torch.cuda.memory_reserved(0) / 1e9
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                vram_str = f"| VRAM: {vram:.1f}/{vram_total:.1f} GB"
            else:
                vram_str = ""
            print(
                f"Epoch {epoch} | Step {step:>4}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | LR: {lr_now:.2e} {vram_str}"
            )

    avg_loss = total_loss / len(loader)
    print(f"\n✓ Epoch {epoch} complete — avg loss: {avg_loss:.4f}\n")

    # Checkpoint
    ckpt_path = f"checkpoints_ft/tiny_gpt_ft_epoch_{epoch}.pth"
    tmp_path  = ckpt_path + ".tmp"
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict'   : scaler.state_dict(),
        'loss'                : avg_loss,
        'csv_path'            : csv_path,
        'pretrained_from'     : PRETRAINED_CHECKPOINT,
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Sample answer every epoch
    print(f"\n[Sample Q&A — epoch {epoch}]")
    try:
        # Grab the first question text from the CSV for a real test
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            sample_row = list(csv.DictReader(f))[0]
            sample_q   = sample_row[QUESTION_COL].strip()
        print(f"Q: {sample_q}")
        print(f"A: {answer(model, enc, sample_q, device, max_new_tokens=80)}")
    except Exception as e:
        print(f"(Sample generation skipped: {e})")
    print("-" * 60)

# ── Final save ─────────────────────────────────────────────────────────────────
torch.save({'model': model.state_dict()}, 'tiny_gpt_finetuned.pth')
print("\nFine-tuned model saved to tiny_gpt_finetuned.pth")
