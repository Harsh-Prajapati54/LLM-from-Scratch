import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self,data,tokonizer,max_length,stride):
        self.inputs = []
        self.targets = []
        
        token_ids = tokonizer.encode(data)