"""
tiny_gpt_chat.py
────────────────
Load a fine-tuned TinyGPT and chat with it in your terminal.

Setup
─────
1. Set MODEL_PATH to your weights file:
      - After fine-tuning : "tiny_gpt_finetuned.pth"
      - From a checkpoint : "checkpoints_ft/tiny_gpt_ft_epoch_4.pth"
2. Run: python tiny_gpt_chat.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import math
import os

# ─────────────────────────────────────────────────────────────────
# CONFIG  ← only thing you need to change
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "tiny_gpt_finetuned.pth"   # path to your .pth file

# Must match the architecture used during training
VOCAB_SIZE  = 50257
D_MODEL     = 128
N_HEADS     = 8
N_LAYERS    = 6
BLOCK_SIZE  = 128
DROPOUT     = 0.0   # always 0 at inference

# Generation settings (tweak these to change output style)
TEMPERATURE = 0.7   # lower = more focused, higher = more creative (0.1 – 1.0)
TOP_K       = 40    # only sample from the top-K most likely tokens
MAX_TOKENS  = 200   # max answer length in tokens
# ─────────────────────────────────────────────────────────────────


EOT_TOKEN = 50256   # <|endoftext|> in GPT-2 vocab — stop token


# ── Model (identical to training) ─────────────────────────────────
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
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
            )
        else:
            scale = math.sqrt(self.head_dim)
            att   = (q @ k.transpose(-2, -1)) / scale
            att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att   = F.softmax(att, dim=-1)
            out   = att @ v

        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


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
            *[TransformerBlock(d_model, n_heads, block_size, dropout)
              for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.ln_f(self.blocks(x))
        return self.head(x)


# ── Weight loader ──────────────────────────────────────────────────
def load_model(path: str, device: torch.device) -> TinyGPT:
    abs_path = os.path.abspath(path)

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"\nModel file not found: {abs_path}\n"
            f"Set MODEL_PATH at the top of this script to the correct path.\n"
            f"Expected one of:\n"
            f"  tiny_gpt_finetuned.pth\n"
            f"  checkpoints_ft/tiny_gpt_ft_epoch_<N>.pth\n"
            f"  checkpoints/tiny_gpt_epoch_<N>.pth"
        )

    print(f"Loading weights from: {abs_path}")
    raw = torch.load(abs_path, map_location=device, weights_only=True)

    # Handle both checkpoint dicts and plain weight dicts
    if "model_state_dict" in raw:
        state = raw["model_state_dict"]
        print(f"  Source: training checkpoint  "
              f"(epoch {raw.get('epoch','?')}, loss {raw.get('loss','?')})")
    elif "model" in raw:
        state = raw["model"]
        print(f"  Source: final save (tiny_gpt_finetuned.pth)")
    else:
        # Bare state dict (no wrapper key)
        state = raw
        print(f"  Source: bare state dict")

    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device    : {device}\n")
    return model


# ── Generation ─────────────────────────────────────────────────────
@torch.no_grad()
def generate_answer(
    model: TinyGPT,
    enc,
    question: str,
    device: torch.device,
    max_tokens: int   = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_k: int        = TOP_K,
) -> str:
    prompt = f"Q: {question.strip()}\nA:"
    idx    = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        # Crop to block_size context window
        logits   = model(idx[:, -BLOCK_SIZE:])
        logits   = logits[:, -1, :] / temperature

        # Top-k filtering
        v, _     = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, -1, None]] = -float("Inf")

        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if next_tok.item() == EOT_TOKEN:
            break

        idx = torch.cat([idx, next_tok], dim=1)

    full     = enc.decode(idx[0].tolist())
    answer   = full.split("\nA:", 1)[-1].strip()
    return answer


# ── Chat loop ──────────────────────────────────────────────────────
def chat(model, enc, device):
    print("=" * 55)
    print("  TinyGPT Chat")
    print(f"  temperature={TEMPERATURE}  top_k={TOP_K}  max_tokens={MAX_TOKENS}")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'settings' to adjust generation parameters.")
    print("=" * 55)

    # Allow live adjustment of generation settings
    temp    = TEMPERATURE
    top_k   = TOP_K
    max_tok = MAX_TOKENS

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "settings":
            print(f"\nCurrent settings:")
            print(f"  temperature = {temp}    (0.1 = focused, 1.0 = creative)")
            print(f"  top_k       = {top_k}   (number of candidate tokens)")
            print(f"  max_tokens  = {max_tok} (max answer length)")
            try:
                t = input("New temperature (Enter to keep): ").strip()
                k = input("New top_k       (Enter to keep): ").strip()
                m = input("New max_tokens  (Enter to keep): ").strip()
                if t: temp    = float(t)
                if k: top_k   = int(k)
                if m: max_tok = int(m)
            except ValueError:
                print("Invalid value — keeping previous settings.")
            print(f"Updated: temperature={temp}, top_k={top_k}, max_tokens={max_tok}")
            continue

        print("Bot: ", end="", flush=True)
        response = generate_answer(
            model, enc, user_input, device,
            max_tokens=max_tok,
            temperature=temp,
            top_k=top_k,
        )
        print(response)


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"GPU  : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU (no GPU detected).")

    enc   = tiktoken.get_encoding("gpt2")
    model = load_model(MODEL_PATH, device)

    chat(model, enc, device)
