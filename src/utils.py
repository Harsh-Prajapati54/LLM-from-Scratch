# it contains an utility functions for the model
import model
from model_config import *
from model import *

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
 context_size=  Gpt_config["context_length"]
)
