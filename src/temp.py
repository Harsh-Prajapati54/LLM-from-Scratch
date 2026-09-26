import statistics

import tiktoken
from datasets import load_from_disk

tokenizer = tiktoken.get_encoding("gpt2")


def token_lengths(stories):
    lengths = []
    for story in stories:
        ids = tokenizer.encode(story)
        lengths.append(len(ids))
    return lengths


def report(lengths):
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)

    def percentile(p):
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        return lengths_sorted[idx]

    print(f"Number of stories: {n}")
    print(f"Min length:        {min(lengths_sorted)}")
    print(f"Max length:        {max(lengths_sorted)}")
    print(f"Mean length:       {statistics.mean(lengths_sorted):.1f}")
    print(f"Median length:     {statistics.median(lengths_sorted):.1f}")
    print(f"90th percentile:   {percentile(90)}")
    print(f"95th percentile:   {percentile(95)}")
    print(f"99th percentile:   {percentile(99)}")

    # Suggest a context length: smallest power-of-2-ish bucket covering 95th percentile
    p95 = percentile(95)
    for candidate in [64, 128, 256, 512, 1024]:
        if candidate >= p95:
            print(f"\nSuggested context_length: {candidate} (covers 95% of stories without truncation)")
            break
    else:
        print(f"\n95th percentile ({p95}) exceeds 1024 — consider context_length=1024 and accept truncation on outliers.")


if __name__ == "__main__":
    dataset = load_from_disk(r"C:\LLM from Scratch\Data")

    # This is the list of individual stories -- one string per story --
    # BEFORE any "".join(...) collapses them into one blob.
    train_stories = dataset["train"]["text"][:200000]
    val_stories = dataset["validation"]["text"][:21990]

    print("=== Train set ===")
    report(token_lengths(train_stories))

    print("\n=== Validation set ===")
    report(token_lengths(val_stories))