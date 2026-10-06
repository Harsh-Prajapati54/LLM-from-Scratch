# this file contains the utility functions for the model, including the attention mechanism and the feedforward network
import torch.nn.functional as F
from torch import nn

from attention_mech import *
from model_config import *


# Activation Functions
class GELU(nn.Module):
    """GELU activation (tanh approximation, as used in GPT-2)."""
    def forward(self,x):
        return F.gelu(x,approximate='tanh') 
    
    
# Feedforward Network - this is a simple two-layer feedforward network with a GELU activation function and dropout for regularization
class FeedForward(nn.Module):
    
    """Position-wise feed-forward block: Linear -> GELU -> Linear -> Dropout.

    Args:
        emb_dim: size of the embedding dimension.
        dropout: dropout probability applied to the output.
    """
    
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        
    def forward(self,x):
        return self.net(x)
    
    
    
class LayerNormalization(nn.Module):
    
    """Layer normalization over the last dimension.

    Args:
        emb_dim: size of the embedding dimension.
        eps: small value added to the variance for numerical stability.
    """
    
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=eps)
        
    def forward(self,x):
        return self.norm(x)
    
    
class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block.

    Args:
        emb_dim: embedding size.
        num_heads: number of attention heads.
        dropout: dropout probability (attention weights and both residual paths).
        qkv_bias: whether q/k/v projections use a bias.
    """
    def __init__(self, emb_dim, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.attention = Attention_Mechanism(emb_dim, emb_dim, context_length, dropout, num_heads, qkv_bias)
        self.feedforward = FeedForward(emb_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        # Attention block
        attn_output = self.attention(self.norm1(x))
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Feedforward block
        ff_output = self.feedforward(self.norm2(x))
        x = x + self.dropout(ff_output)  # Residual connection
        
        return x
    
    
if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    B, T, D, H, CTX = 2, 16, 768, 12, 1024   # batch, tokens, emb_dim, heads, context_length

    x = torch.randn(B, T, D)

    # 1. GELU: output shape matches input
    assert GELU()(x).shape == x.shape
    print("GELU ok")

    # 2. FeedForward: shape stays (B, T, D)
    ff = FeedForward(D)
    assert ff(x).shape == x.shape
    print("FeedForward ok")

    # 3. LayerNormalization: each token ends up with mean ~0 and std ~1
    ln = LayerNormalization(D)
    out = ln(x)
    assert out.shape == x.shape
    assert torch.allclose(out.mean(-1), torch.zeros(B, T), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(B, T), atol=1e-3)
    print("LayerNormalization ok")

    # 4. TransformerBlock: shape stays (B, T, D)
    block = TransformerBlock(D, H, CTX, dropout=0.1)
    assert block(x).shape == x.shape
    print("TransformerBlock shape ok")

    # 5. Causal test: changing the last token must not change earlier outputs
    block.eval()                      # turn dropout off so results are deterministic
    with torch.no_grad():
        y1 = block(x)
        x2 = x.clone()
        x2[:, -1, :] = torch.randn(B, D)   # change only the last token
        y2 = block(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6), "Model is looking at the future!"
    assert not torch.allclose(y1[:, -1], y2[:, -1]), "Last token output should change"
    print("Causal mask ok")

    # 6. Backward pass: every parameter should get a gradient
    block.train()
    block(x).sum().backward()
    missing = [n for n, p in block.named_parameters() if p.grad is None]
    assert not missing, f"No gradient for: {missing}"
    print("Gradients ok")

    # 7. Parameter count
    n_params = sum(p.numel() for p in block.parameters())
    print(f"Block parameters: {n_params:,}")