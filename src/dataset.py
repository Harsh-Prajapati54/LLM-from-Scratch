from datasets import load_from_disk
# loads a dataset from disk 
dataset = load_from_disk(r"C:\LLM from Scratch\Data")

# joining all row as an single text 

dataset_text = " ".join(dataset["train"]['text'][:200000])

data = dataset_text

print(f"data sample:  {data[:500]}")