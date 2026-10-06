# this file contains the utility functions for the model, including the attention mechanism and the feedforward network
from torch import nn
import torch.nn.functional as F

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