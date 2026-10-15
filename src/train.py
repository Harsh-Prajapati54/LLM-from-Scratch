# training script for the model
import time

import tiktoken
import torch
from Dataloader import train_loader, val_loader
from model import model
from model import *
from utils import calc_loss_batch, calc_loss_loader, generate_and_print_sample


# Optimizer and training parameters
optimizer = torch.optim.AdamW(model.parameters(),
                            lr=1e-4,
                            weight_decay=0.1)



class Trainer:
    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter
            )
        model.train()

        return train_loss, val_loss
    

        
    def train_model(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq ,eval_iter,start_context,tokenizer):
        
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        n_params = sum(p.numel() for p in model.parameters())
        
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        
        print("=" * 60)
        print("Training started")
        print(f"  Device          : {device}")
        print(f"  Parameters      : {n_params / 1e6:.1f}M")
        print(f"  Epochs          : {num_epochs}")
        print(f"  Batch size      : {train_loader.batch_size}")
        print(f"  Batches / epoch : {steps_per_epoch} train, {len(val_loader)} val")
        print(f"  Total steps     : {total_steps}")
        print(f"  Evaluate every  : {eval_freq} steps (using {eval_iter} batches)")
        print(f"  Learning rate   : {optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()

            for batch_idx, (input_batch, target_batch) in enumerate(train_loader,start=1):
                
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)

                loss.backward()
                optimizer.step()

                tokens_seen += input_batch.numel()
                global_step += 1

                if (global_step + 1) % eval_freq == 0:
                    train_loss, val_loss = Trainer.evaluate_model(
                        model,
                        train_loader,
                        val_loader,
                        device,
                        eval_iter
                    )

                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f} | "
                        f"step {global_step + 1}/{total_steps}, "
                        f"{tokens_seen:,} tokens, "
                        f"{time.time() - start_time:.0f}s elapsed",
                        flush=True
                    )
                    
            print(f"Epoch {epoch+1}/{num_epochs} finished in {time.time() - epoch_start:.1f}s "
                f"(last batch loss {loss.item():.3f})")
            print("Generating sample text...")
            
            
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context
            )
            
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training finished")
        print(f"  Total time      : {total_time / 60:.1f} min")
        print(f"  Tokens seen     : {tokens_seen:,}")
        if val_losses:
            best = min(val_losses)
            print(f"  Best val loss   : {best:.3f} (evaluation #{val_losses.index(best) + 1})")
            print(f"  Final val loss  : {val_losses[-1]:.3f}")
        print("=" * 60)
        return train_losses, val_losses, track_tokens_seen



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 5
    eval_freq = 5
    eval_iter = 2
    start_context = "Once upon a time"
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_losses, val_losses, track_tokens_seen = Trainer.train_model(model,
                                                                    train_loader,
                                                                    val_loader,
                                                                    optimizer,
                                                                    device,
                                                                    num_epochs = num_epochs,eval_freq = eval_freq,
                                                                    eval_iter = eval_iter,
                                                                    start_context = start_context,tokenizer=tokenizer)