import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self,data,tokonizer,max_length,stride):
        self.inputs = []
        self.targets = []
        
        token_ids = tokonizer.encode(data)
        for i in range(0,len(token_ids)-max_length,stride):
            input_seq = token_ids[i:i+max_length]
            target_seq = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_seq))
            self.target_ids.append(torch.tensor(target_seq))
    def __len__(self):
            return len(self.input_ids)
        
    def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
        
    def create_dataloader_v1(data, batch_size = 32,
                         max_length =256,
                         stride = 128,
                         drop_last = True,
                         shuffle = True,
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")  # Initializes the  tokenizer
    dataset = GPTDataset(data, tokenizer, max_length, stride)    # Creates dataset
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            drop_last=drop_last,   # Drops the last batch if it is smaller than the specified batch size to prevent loss spike
                            num_workers=num_workers)
    return dataloader