import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader

raw_text = """
        Build a Large Language Model (From Scratch) was written to help you understand and
        create your own GPT-like large language models (LLMs) from the ground up. It
        begins by focusing on the fundamentals of working with text data and coding atten
        tion mechanisms and then guides you through implementing a complete GPT
        model from scratch. The book then covers the pretraining mechanism as well as
        fine-tuning for specific tasks such as text classification and following instructions. By
        the end of this book, you’ll have a deep understanding of how LLMs work and the
        skills to build your own models. While the models you’ll create are smaller in scale
        compared to the large foundational models, they use the same concepts and serve
        as powerful educational tools to grasp the core mechanisms and techniques used in
        building state-of-the-art LLMs.

        """
        
vocab_size = 50257
output_dim = 256
max_len = 1024
context_length = max_len

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
    encoded_text = tokenizer.encode(raw_text)
 
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(context_length, output_dim)
 
    max_length = 4
    dataloader = train_dataloader(
        raw_text, batch_size=8, max_length=max_length, stride=max_length
    )
 
    for batch_inputs, batch_targets in dataloader:
        token_embeds = token_embedding_layer(batch_inputs)
        pos_embeds = pos_embedding_layer(torch.arange(max_length))
        input_embeddings = token_embeds + pos_embeds
 
        print("Input shape:", batch_inputs.shape)
        print("Embedding shape:", input_embeddings.shape)
        break