# it contains an utility functions for the model
import model
from model_config import *
from model import *
from src.train import *
import torch
import tiktoken

def generate_text(model,idx,max_new_tokens,context_size):
    """Generate text from a language model using greedy decoding.

    Repeatedly predicts the most likely next token and appends it to the
    sequence. At each step, only the last `context_size` tokens are fed to
    the model, so longer inputs are cropped from the left.

    Args:
        model: a GPT-style model that returns logits of shape
            (batch, tokens, vocab_size).
        idx: tensor of token IDs with shape (batch, tokens), used as the
            starting prompt.
        max_new_tokens: number of new tokens to generate.
        context_size: maximum number of tokens the model can see at once
            (the model's context_length).

    Returns:
        Tensor of shape (batch, tokens + max_new_tokens) containing the
        original prompt followed by the generated tokens.

    Note:
        Call model.eval() before using this function so dropout is off.
        Decoding is greedy (argmax), so the same prompt always gives the
        same output.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:-context_size:]
        with torch.no_grad():
            logits = model(idx_cond.unsqueeze(0))
            
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        
    return idx


def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|'})
    encoded_tensor  = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=Gpt_config["context_length"]
)
def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
                
            logits = logits[:,-1,:]
            probas = torch.softmax(logits,dim=1)
            idx_next = torch.argmax(probas,dim = 1,keepdim = True)
            idx = torch.cat((idx,idx_next),dim = 1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
    
    def calc_loss_batch(input_batch,target_batch,model,device):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(torch.device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    
        return loss 

    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))  
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(
                    input_batch, target_batch, model, torch.device
                )
                total_loss += loss.item()
            else:
                break
        
        return total_loss / num_batches