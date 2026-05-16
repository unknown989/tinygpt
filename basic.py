import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
<<<<<<< HEAD
import time
from datetime import datetime, timedelta
=======

# ── P100-compatible precision settings (new API, avoids deprecation warnings) ──
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.benchmark = True
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e

enc = tiktoken.get_encoding("gpt2")

cpu_tensor = torch.tensor([1, 2, 3])
print(f"CPU Tensor Device: {cpu_tensor.device}")

if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"GPU Tensor Device: {gpu_tensor.device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    CUDA_CAPABILITY = torch.cuda.get_device_capability(0)
    print(f"CUDA Capability: {CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]}")
else:
    print("CUDA not available, all tensors on CPU.")
    CUDA_CAPABILITY = (0, 0)

# ── Hyperparameters ────────────────────────────────────────────────────────────
block_size        = 128
BATCH_SIZE        = 128
GRAD_ACCUM_STEPS  = 4
NUM_WORKERS       = 4
LEARNING_RATE     = 3e-4
NUM_EPOCHS        = 10


# ── Dataset path configuration ────────────────────────────────────────────────
# Set this to the path of your .txt file before running.
# Examples:
#   DATASET_PATH = "/kaggle/input/my-dataset/data.txt"
#   DATASET_PATH = "./my_data.txt"
#   DATASET_PATH = "/content/drive/MyDrive/data.txt"  # Google Colab
DATASET_PATH = ""   # <-- SET YOUR PATH HERE

def resolve_dataset_path(path: str) -> str:
    """
    Resolve the dataset path. If DATASET_PATH is empty, prompts the user
    interactively for a path and validates it exists.
    """
    if path and os.path.isfile(path):
        return path

    if path:
        print(f"WARNING: DATASET_PATH '{path}' does not exist or is not a file.")

    print("\n--- Dataset Selection ---")
    print("No valid DATASET_PATH was set at the top of this script.")
    while True:
        user_path = input("Enter the full path to your .txt dataset file: ").strip()
        if os.path.isfile(user_path):
            return user_path
        print(f"  File not found: '{user_path}'. Please try again.")


def load_text(path: str) -> str:
    """Load and return the full text from a .txt file."""
    abs_path = os.path.abspath(path)
    size_mb  = os.path.getsize(abs_path) / 1e6
    print(f"Loading dataset : {abs_path}")
    print(f"File size       : {size_mb:.2f} MB")
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(f"Characters      : {len(text):,}")
    return text


# ── Model ──────────────────────────────────────────────────────────────────────
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
                q, k, v,
                attn_mask=None,
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


<<<<<<< HEAD
block_size = 128

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
=======
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e
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


<<<<<<< HEAD
def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

def estimate_remaining_time(elapsed_time, steps_completed, total_steps):
    """Estimate remaining time based on elapsed time and progress"""
    if steps_completed == 0:
        return "Calculating..."
    
    time_per_step = elapsed_time / steps_completed
    remaining_steps = total_steps - steps_completed
    remaining_seconds = time_per_step * remaining_steps
    
    return format_time(remaining_seconds)

model = TinyGPT(50257, d_model=128, n_heads=8, n_layers=6, block_size=128, dropout=0.1)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
=======
# ── Helpers ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, enc, prompt, device, max_new_tokens=200, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :] / temperature
        v, _      = torch.topk(logits, top_k)
        logits[logits < v[:, -1, None]] = -float("Inf")
        probs     = F.softmax(logits, dim=-1)
        idx_next  = torch.multinomial(probs, num_samples=1)
        idx       = torch.cat((idx, idx_next), dim=1)
    return enc.decode(idx[0].tolist())
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e

@torch.no_grad()
def generate(model, enc, prompt, device, max_new_tokens=200, temperature=0.8, top_k=50):
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

def get_available_checkpoints():
<<<<<<< HEAD
    """Get list of available checkpoints in the checkpoints directory"""
    checkpoint_files = glob.glob("checkpoints/tiny_gpt_epoch_*.pth")
    # Extract epoch numbers and sort
=======
    checkpoint_files = glob.glob("checkpoints/tiny_gpt_epoch_*.pth")
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_epoch_')[-1].split('.pth')[0])
            epochs.append(epoch)
<<<<<<< HEAD
        except:
            continue
    return sorted(epochs)

def ask_user_for_checkpoint():
    """Ask user whether to start from scratch or from a checkpoint"""
    print("\n--- Training Options ---")
    print("Do you want to:")
    print("1. Start training from scratch")
    print("2. Resume from a specific epoch checkpoint")
    print("3. Test generation with existing model (if available)")
    
=======
        except Exception:
            continue
    return sorted(epochs)


def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None):
    abs_path = os.path.abspath(path)
    print(f"\n--- Checkpoint Diagnostics ---")
    print(f"CWD          : {os.getcwd()}")
    print(f"Resolved path: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"ERROR: File does not exist at {abs_path}")
        if os.path.isdir('checkpoints'):
            print(f"Files in checkpoints/: {os.listdir('checkpoints')}")
        else:
            print("ERROR: 'checkpoints' directory does not exist.")
        return None, False

    file_size = os.path.getsize(abs_path)
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    if file_size < 1024:
        print(f"ERROR: File too small ({file_size} bytes) — likely corrupt or empty.")
        return None, False

    print("Attempting torch.load ...")
    try:
        checkpoint = torch.load(abs_path, map_location=device, weights_only=True)
    except RuntimeError as e:
        print(f"\nERROR: torch.load failed — {e}")
        return None, False
    except Exception as e:
        print(f"\nUnexpected error during torch.load: {type(e).__name__}: {e}")
        return None, False

    print(f"torch.load succeeded. Keys: {list(checkpoint.keys())}")

    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded from 'model_state_dict'")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Model state loaded from 'model'")
        else:
            print(f"ERROR: No model weights found. Keys: {list(checkpoint.keys())}")
            return None, False
    except RuntimeError as e:
        print(f"ERROR loading model weights: {e}")
        return None, False

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state restored.")
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state restored.")
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("GradScaler state restored.")

    if 'loss'  in checkpoint: print(f"Saved loss : {checkpoint['loss']:.4f}")
    if 'epoch' in checkpoint: print(f"Saved epoch: {checkpoint['epoch']}")
    print("--- End Diagnostics ---\n")
    return checkpoint, True


def ask_user_for_checkpoint():
    print("\n--- Training Options ---")
    print("1. Start training from scratch")
    print("2. Resume from a specific epoch checkpoint")
    print("3. Test generation with existing model")
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")

<<<<<<< HEAD
# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Check if there are any checkpoints available
checkpoint_epochs = get_available_checkpoints()
has_checkpoints = len(checkpoint_epochs) > 0

if has_checkpoints:
    print(f"Found existing checkpoints for epochs: {checkpoint_epochs}")
    choice = ask_user_for_checkpoint()
    
    if choice == 1:
        # Start from scratch
        print("Starting training from scratch...")
        start_epoch = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
        
    elif choice == 2:
        # Resume from specific epoch
        while True:
            try:
                epoch_choice = int(input(f"Enter the epoch number to resume from {checkpoint_epochs}: "))
                if epoch_choice in checkpoint_epochs:
                    checkpoint_path = f"checkpoints/tiny_gpt_epoch_{epoch_choice}.pth"
                    print(f"Loading checkpoint from {checkpoint_path}...")
                    
                    checkpoint = torch.load(checkpoint_path)
                    
                    # Load model state
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint['model'])
                    
                    # Initialize optimizer and scheduler
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
                    
                    # Load optimizer and scheduler states if available
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    start_epoch = epoch_choice + 1  # Start from next epoch
                    print(f"Resuming training from epoch {start_epoch}")
                    break
                else:
                    print(f"Invalid epoch. Please choose from {checkpoint_epochs}")
            except ValueError:
                print("Please enter a valid number")
            except FileNotFoundError:
                print("Checkpoint file not found. Please try again.")
                
    elif choice == 3:
        # Test generation with existing model
        try:
            # Try to load the latest checkpoint
            latest_epoch = max(checkpoint_epochs)
            checkpoint_path = f"checkpoints/tiny_gpt_epoch_{latest_epoch}.pth"
            checkpoint = torch.load(checkpoint_path)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model'])
            
            print(f"Loaded model from epoch {latest_epoch}")
            
            # Test generation
            prompt = input("Enter a prompt for text generation (default: 'Once upon a time'): ").strip()
            if not prompt:
                prompt = "Once upon a time"
            
            generated_text = generate(model, enc, prompt, device, max_new_tokens=50)
            print("\nGenerated text:")
            print(generated_text)
            
            # Ask if they want to continue training
            continue_training = input("\nDo you want to continue training? (y/n): ").strip().lower()
            if continue_training == 'y':
                choice = 2  # Force resume from checkpoint
                # Re-initialize optimizer and scheduler for training
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
                start_epoch = latest_epoch + 1
            else:
                print("Exiting...")
                exit()
        except Exception as e:
            print(f"Error loading model for generation: {e}")
            print("Starting training from scratch instead...")
            choice = 1
            start_epoch = 0
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
else:
    print("No existing checkpoints found. Starting training from scratch...")
    start_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

# Load and prepare data
print("Loading data...")
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

print("Training...")
model.train()
max_step = len(loader)

# Create checkpoint directory if it doesn't exist
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Calculate total training steps
total_epochs = 10
remaining_epochs = total_epochs - start_epoch
total_steps_remaining = remaining_epochs * max_step

print(f"\n{'='*60}")
print(f"TRAINING CONFIGURATION:")
print(f"Total epochs: {total_epochs}")
print(f"Starting from epoch: {start_epoch}")
print(f"Epochs remaining: {remaining_epochs}")
print(f"Steps per epoch: {max_step}")
print(f"Total steps remaining: {total_steps_remaining}")
print(f"{'='*60}\n")

# Training loop with time estimation
training_start_time = time.time()
epoch_times = []  # Store time taken for each completed epoch

for epoch in range(start_epoch, total_epochs):
    epoch_start_time = time.time()
    total_loss = 0
    
    print(f"\n--- Epoch {epoch}/{total_epochs-1} ---")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    for step, (x, y) in enumerate(loader):
        step_start_time = time.time()
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        
        # Calculate and display progress with time estimates
        if step % 100 == 0:
            current_time = time.time()
            epoch_elapsed = current_time - epoch_start_time
            total_elapsed = current_time - training_start_time
            
            # Steps completed in this epoch and overall
            steps_in_epoch = step + 1
            steps_completed_total = (epoch - start_epoch) * max_step + steps_in_epoch
            
            # Time estimates
            epoch_remaining = estimate_remaining_time(epoch_elapsed, steps_in_epoch, max_step)
            total_remaining = estimate_remaining_time(total_elapsed, steps_completed_total, total_steps_remaining)
            
            # Estimated completion time
            if total_remaining != "Calculating...":
                try:
                    # Parse the remaining time string to get seconds
                    remaining_str = total_remaining
                    if 's' in remaining_str:
                        remaining_seconds = float(remaining_str.replace('s', ''))
                    elif 'm' in remaining_str:
                        remaining_seconds = float(remaining_str.replace('m', '')) * 60
                    elif 'h' in remaining_str:
                        remaining_seconds = float(remaining_str.replace('h', '')) * 3600
                    else:
                        remaining_seconds = 0
                    
                    eta = datetime.now() + timedelta(seconds=remaining_seconds)
                    eta_str = eta.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    eta_str = "Calculating..."
            else:
                eta_str = "Calculating..."
            
            print(f"Epoch {epoch}, Step {step}/{max_step} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Epoch Progress: {steps_in_epoch/max_step*100:.1f}% | "
                  f"Epoch ETA: {epoch_remaining} | "
                  f"Total ETA: {total_remaining} | "
                  f"Est. Completion: {eta_str}")
    
    # Epoch completed
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_duration)
    
    avg_loss = total_loss / len(loader)
    print(f"\n✓ Epoch {epoch} completed!")
    print(f"  Duration: {format_time(epoch_duration)}")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    # Estimate remaining time based on average epoch time
    if len(epoch_times) > 0:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        epochs_left = total_epochs - epoch - 1
        estimated_time_left = avg_epoch_time * epochs_left
        
        print(f"  Average epoch time: {format_time(avg_epoch_time)}")
        print(f"  Estimated time remaining: {format_time(estimated_time_left)}")
        
        if epochs_left > 0:
            estimated_completion = datetime.now() + timedelta(seconds=estimated_time_left)
            print(f"  Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save checkpoint at the end of each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f'tiny_gpt_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Optional: Generate sample text after each epoch
    if epoch % 2 == 0:  # Every 2 epochs
        sample_prompt = "To be or not to be"
        generated = generate(model, enc, sample_prompt, device, max_new_tokens=50)
        print(f"\n  Sample generation (epoch {epoch}):")
        print(f"  {generated[:200]}...")  # Truncate for display
        print("-" * 50)

# Training completed
total_training_time = time.time() - training_start_time
print(f"\n{'='*60}")
print(f"TRAINING COMPLETED!")
print(f"Total training time: {format_time(total_training_time)}")
print(f"Average time per epoch: {format_time(total_training_time / remaining_epochs)}")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# Also save the final model in the original location for compatibility
torch.save({
    'model': model.state_dict(), 
    'optimizer': optimizer.state_dict(), 
    'scheduler': scheduler.state_dict()
}, 'tiny_gpt.pth')
print("Final checkpoint saved successfully.")
=======

def make_optimizer_and_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
    return optimizer, scheduler


# ── Device setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = TinyGPT(
    vocab_size=50257,
    d_model=128,
    n_heads=8,
    n_layers=6,
    block_size=block_size,
    dropout=0.1,
)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
model = model.to(device)

if CUDA_CAPABILITY[0] >= 7 and hasattr(torch, 'compile'):
    print("Compiling model with torch.compile ...")
    model = torch.compile(model)
else:
    print(
        f"torch.compile skipped — device is CUDA capability "
        f"{CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]} (needs ≥ 7.0)"
    )

scaler = torch.amp.GradScaler('cuda')


# ── Checkpoint / training mode selection ───────────────────────────────────────
checkpoint_epochs = get_available_checkpoints()
has_checkpoints   = len(checkpoint_epochs) > 0

if has_checkpoints:
    print(f"Found checkpoints for epochs: {checkpoint_epochs}")
    choice = ask_user_for_checkpoint()
else:
    print("No existing checkpoints found. Starting from scratch.")
    choice = 1

if choice == 1:
    print("Starting training from scratch ...")
    start_epoch           = 0
    optimizer, scheduler  = make_optimizer_and_scheduler(model)

elif choice == 2:
    while True:
        try:
            epoch_choice = int(input(f"Resume from epoch {checkpoint_epochs}: "))
            if epoch_choice not in checkpoint_epochs:
                print(f"Invalid. Choose from {checkpoint_epochs}")
                continue

            optimizer, scheduler = make_optimizer_and_scheduler(model)
            checkpoint, success  = load_checkpoint(
                f"checkpoints/tiny_gpt_epoch_{epoch_choice}.pth",
                model, device, optimizer, scheduler, scaler,
            )

            if not success:
                retry = input("Load failed. Try another epoch? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Falling back to training from scratch.")
                    start_epoch          = 0
                    optimizer, scheduler = make_optimizer_and_scheduler(model)
                continue

            start_epoch = epoch_choice + 1
            print(f"Resuming from epoch {start_epoch}")
            break

        except ValueError:
            print("Please enter a valid number.")

elif choice == 3:
    latest_epoch = max(checkpoint_epochs)
    checkpoint, success = load_checkpoint(
        f"checkpoints/tiny_gpt_epoch_{latest_epoch}.pth",
        model, device,
    )

    if success:
        print(f"Loaded model from epoch {latest_epoch}")
        prompt = input("Enter a prompt (default: 'Once upon a time'): ").strip()
        max_new_tokens = input("Max tokens (default: 50): ").strip()
        if not prompt:
            prompt = "Once upon a time"
        max_new_tokens = int(max_new_tokens) if max_new_tokens else 50

        print("\nGenerated text:")
        print(generate(model, enc, prompt, device, max_new_tokens))

        if input("\nContinue training? (y/n): ").strip().lower() == 'y':
            optimizer, scheduler = make_optimizer_and_scheduler(model)
            if checkpoint:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = latest_epoch + 1
        else:
            import sys
            print("Exiting.")
            sys.exit(0)
    else:
        print("Could not load model. Starting from scratch.")
        start_epoch          = 0
        optimizer, scheduler = make_optimizer_and_scheduler(model)


# ── Data ───────────────────────────────────────────────────────────────────────
print("\nLoading data ...")

# Resolve and load the dataset — edit DATASET_PATH at the top of this file,
# or leave it empty to be prompted interactively at runtime.
dataset_path = resolve_dataset_path(DATASET_PATH)
text         = load_text(dataset_path)
tokens       = enc.encode(text)
print(f"Tokens          : {len(tokens):,}")


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens     = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        assert len(tokens) > block_size, \
            f"Text too short: {len(tokens)} tokens, need > {block_size}."

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx:     idx + self.block_size]
        y = self.tokens[idx + 1: idx + self.block_size + 1]
        return x, y


dataset = TextDataset(tokens, block_size)
loader  = DataLoader(
    dataset,
    batch_size         = BATCH_SIZE,
    shuffle            = True,
    num_workers        = NUM_WORKERS,
    pin_memory         = True,
    prefetch_factor    = 2,
    persistent_workers = True,
)

print(f"Dataset        : {len(dataset):,} samples")
print(f"Batches/epoch  : {len(loader):,}")
print(f"Effective batch: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")


# ── Training loop ──────────────────────────────────────────────────────────────
print("\nTraining ...")
os.makedirs("checkpoints", exist_ok=True)
max_step = len(loader)

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            _, loss  = model(x, y)
            loss_acc = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss_acc).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if step % 100 == 0:
            vram_used  = torch.cuda.memory_reserved(0) / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(
                f"Epoch {epoch} | Step {step:>5}/{max_step} | "
                f"Loss: {loss.item():.4f} | "
                f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB"
            )

    avg_loss = total_loss / max_step
    print(f"\nEpoch {epoch} complete — avg loss: {avg_loss:.4f}\n")

    ckpt_path = f"checkpoints/tiny_gpt_epoch_{epoch}.pth"
    tmp_path  = ckpt_path + ".tmp"
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict'   : scaler.state_dict(),
        'loss'                : avg_loss,
        'dataset_path'        : dataset_path,   # record which dataset was used
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    if epoch % 2 == 0:
        sample = generate(model, enc, "Once upon a time", device, max_new_tokens=50)
        print(f"Sample (epoch {epoch}):\n{sample}")
        print("-" * 50)

torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, 'tiny_gpt.pth')
print("Final model saved to tiny_gpt.pth")
>>>>>>> af31e2f748080903a18e75ad0e89c1d056e0906e
