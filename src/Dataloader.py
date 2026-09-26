import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import load_dataset

train_data, val_data = load_dataset()

        
vocab_size = 50257
output_dim = 512 
max_len = 512
context_length = max_len # Context length (context window) is how many tokens the model can "see" at once when predicting the next token.

torch.manual_seed(20)

class GPTDataset(Dataset):
        """
        Converts raw text into input-target token sequences.

        Input:
                [t1, t2, t3, t4]

        Target:
                [t2, t3, t4, t5]

        The target is shifted by one token because the model
        learns to predict the next token.
        """
    
        def __init__(self,data,tokonizer,max_length,stride):
                self.input_ids = []
                self.target_ids = []
        
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
        
        

def train_dataloader(data,
                         batch_size = 32,
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

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(train_data)
 
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(context_length, output_dim)
 
    max_length = 512
    dataloader = train_dataloader(
        train_data, batch_size=32, max_length=max_length, stride=max_length
    )
 
    for batch_inputs, batch_targets in dataloader:
        token_embeds = token_embedding_layer(batch_inputs)
        pos_embeds = pos_embedding_layer(torch.arange(max_length))
        input_embeddings = token_embeds + pos_embeds
        
        print(len(encoded_text))
        print("Input shape:", batch_inputs.shape)
        print("Embedding shape:", input_embeddings.shape)
        # print("first embedding",input_embeddings[0][0])
        # print("total embeddings :", len(input_embeddings))
        print("Number of batches:", len(dataloader))
        print("Approx total examples:", len(dataloader) * dataloader.batch_size)
        break