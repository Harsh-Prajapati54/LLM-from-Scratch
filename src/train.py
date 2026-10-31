# training script for the model
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



class Evaluate:
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
    

class Trainer:
    def train_model(model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq ,eval_iter,start_context,tokenizer):
        
        
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        
        
        for epoch in range(num_epochs):
            model.train()

            for input_batch, target_batch in train_loader:
                
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)

                loss.backward()
                optimizer.step()

                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step  % eval_freq == 0:
                    train_loss, val_loss = Evaluate.evaluate_model(
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
                    )
                    

            
            
            generate_and_print_sample(
                model,
                tokenizer,
                device,
                start_context
            )
            

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