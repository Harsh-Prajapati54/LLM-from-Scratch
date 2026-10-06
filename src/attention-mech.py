import torch.nn.functional as F
from torch import nn

# implementing attention mechanismin this script

class Attention_Mechanism(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_head"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = dropout
        
        self.W_query = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_key   = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        
    def split_head(self,x):
         # (batch, tokens, d_out) -> (batch, heads, tokens, head_dim)
         
         b, T, _ = x.shape
         return x.view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
     
    def forward(self,x):
        
        b, T, _ = x.shape
        
        # 1. Make q, k, v and split them into heads
        
        q = self.split_head(self.W_query(x))  # (batch, heads, tokens, head_dim)
        k = self.split_head(self.W_key(x))    # (batch, heads, tokens, head_dim)
        v = self.split_head(self.w_value(x))  # (batch, heads, tokens, head_dim)
        
         # 2. Attention (scores, causal mask, softmax, dropout, weighted sum) in one fast call
        context = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p= self.dropout if self.training else 0.0,
            is_causal=True
            )  # (batch, heads, tokens, head_dim)
        
                # 3. Join the heads back together and mix them with the output layer
        context = context.transpose(1, 2).reshape(b, T, self.d_out)  # (batch, tokens, d_out)
        
        return self.out_proj(context)  # (batch, tokens, d_out)