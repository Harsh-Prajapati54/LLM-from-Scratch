# this file contains the utility functions for the model, including the attention mechanism and the feedforward network
import torch.nn.functional as F
from attention_mech import Attention_Mechanism
from torch import nn


# Activation Functions
class GELU(nn.Module):
    def forward(self,x):
        return F.gelu(x,approximate='tanh')  # GELU activation function
    
    
# feedforward network for the transformer block
class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            GELU(),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self,x):
        return self.net(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, eps=eps)
        
    def forward(self,x):
        return self.norm(x)
    
    
class TransformerBlock(nn.Module):
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