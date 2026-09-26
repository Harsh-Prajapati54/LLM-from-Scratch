import torch
import torch.nn as nn

from Dataloader import *

# implementing attention mechanismin this notebook 

class Attention_Mechanism(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_head"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads