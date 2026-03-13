import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import math

enc = tiktoken.get_encoding("gpt2")


block_size = 128
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, n_heads, block_size, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


model = TinyGPT(50257, d_model=128, n_heads=8, n_layers=6, block_size=128, dropout=0.1)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


try:
    
    ckpt = torch.load('tiny_gpt.pth')
    print("Checkpoint loaded successfully.")    
    ckpt = torch.load('tiny_gpt.pth')
    model.load_state_dict(ckpt['model'])

    @torch.no_grad()
    def generate(model, enc, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
        model.eval()
        idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1, None]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return enc.decode(idx[0].tolist())

    prompt = "Once upon a time"
    generated_text = generate(model, enc, prompt, max_new_tokens=50)
    print(generated_text)
except FileNotFoundError:
    print("No checkpoint found. Starting training from scratch.")
    text = open("t8.shakespeare.txt", "r").read()
    tokens = enc.encode(text)
    class TextDataset(Dataset):
        def __init__(self, tokens, block_size):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
            self.block_size = block_size
            assert (
                len(tokens) > block_size
            ), f"Text too short: {len(tokens)} tokens, need > {block_size}. Reduce block_size or use more text."

        def __len__(self):
            return len(self.tokens) - self.block_size

        def __getitem__(self, idx):
            x = self.tokens[idx : idx + self.block_size]
            y = self.tokens[idx + 1 : idx + self.block_size + 1]
            return x, y


    dataset = TextDataset(tokens, block_size)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)


    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model, n_heads, block_size, dropout):
            super().__init__()
            assert d_model % n_heads == 0
            self.n_head = n_heads
            self.head_dim = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model)
            self.drop = nn.Dropout(dropout)
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(C, dim=2)

            def split_heads(tensor):
                return tensor.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            q, k, v = map(split_heads, (q, k, v))
            scale = math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) / scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(out)


    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, block_size, dropout):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x


    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

    model.train()
    for epoch in range(10):
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")


    # save
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, 'tiny_gpt.pth')
    print("Checkpoint saved successfully.")