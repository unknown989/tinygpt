import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import glob
import time
from datetime import datetime, timedelta

enc = tiktoken.get_encoding("gpt2")


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
    """Get list of available checkpoints in the checkpoints directory"""
    checkpoint_files = glob.glob("checkpoints/tiny_gpt_epoch_*.pth")
    # Extract epoch numbers and sort
    epochs = []
    for f in checkpoint_files:
        try:
            epoch = int(f.split('_epoch_')[-1].split('.pth')[0])
            epochs.append(epoch)
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
    
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")

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