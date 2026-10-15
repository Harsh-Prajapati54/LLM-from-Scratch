from datasets import load_dataset as hf_load_dataset


def load_dataset():
    
    # ============================================================
    # Load dataset -- Tiny stories 
    # ============================================================
    train = hf_load_dataset("roneneldan/TinyStories", split="train[:20000]")
    val = hf_load_dataset("roneneldan/TinyStories", split="validation[:2190]")

    # ============================================================
    # Prepare training and validation text
    # ============================================================

    dataset_text_train = "".join(train["text"][:20000])
    dataset_text_val = "".join(val["text"][:2190])
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