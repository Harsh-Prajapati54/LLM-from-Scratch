from datasets import load_from_disk
# loads a dataset from disk 
dataset = load_from_disk(r"C:\LLM from Scratch\Data")

dataset_text = " ".join(dataset["train"]['text'][:200000])

print(f"total text in dataset: {len(dataset_text)}")