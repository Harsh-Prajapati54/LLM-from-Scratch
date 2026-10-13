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
        self.context_length = context_length
        
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
        assert T <= self.context_length, f"Sequence length {T} exceeds context_length {self.context_length}"
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
    

if __name__ == "__main__":
    import torch
    from model_config import Gpt_config as c

    torch.manual_seed(20)
    attn = Attention_Mechanism(c["emb_dim"], c["emb_dim"], c["context_length"],
                               c["drop_rate"], c["num_heads"], c["qkv_bias"]).eval()

    # 1. shape: output must match input
    x = torch.randn(2, c["context_length"], c["emb_dim"])  # (batch, tokens, emb)
    out = attn(x)
    print("input :", tuple(x.shape), "\noutput:", tuple(out.shape))
    assert out.shape == x.shape

    # 2. causal: changing the LAST token must not change any earlier output
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert torch.allclose(out[:, :-1], attn(x2)[:, :-1], atol=1e-5), "future token leaked into the past"
    print("causal check passed")

    # 3. gradients: every weight must receive a gradient
    attn.train()
    attn(x).sum().backward()
    assert all(p.grad is not None for p in attn.parameters()), "a weight got no gradient"
    print("gradient check passed")