from datasets import load_from_disk


def load_dataset():
    
    # ============================================================
    # Load dataset -- Tiny stories 
    # ============================================================
    
    dataset = load_from_disk(r"C:\LLM from Scratch\Data")

    # ============================================================
    # Prepare training and validation text
    # ============================================================

    dataset_text_train = "".join(dataset["train"]['text'][:200000])
    dataset_text_val = "".join(dataset["validation"]['text'][:21990]) 
  
    train_data = dataset_text_train
    val_data= dataset_text_val
    
    return train_data,val_data

if __name__ =="__main__":
    
    train_data , val_data = load_dataset()
    # ============================================================
    # Check loaded data
    # ===========================================================
    print(f"train data sample:\n  {train_data[:500]}")
    print("=" * 100)
    print(f"val data sample :\n {val_data[:500]}")