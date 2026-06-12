"""Data utilities: TinyStories token stream (for training) and C4 text (for analysis marginal/freq-rare)."""
import os
import numpy as np
import torch


def tinystories_tokens(n_tokens=40_000_000, cache=None):
    """Return a uint16 array of GPT-2-BPE TinyStories tokens (cached to disk)."""
    cache = cache or f"/home/claudeuser/ts_tokens_{n_tokens}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    buf, eot = [], (tok.eos_token_id or 50256)
    for ex in ds:
        buf.extend(tok(ex["text"])["input_ids"]); buf.append(eot)
        if len(buf) >= n_tokens:
            break
    arr = np.array(buf[:n_tokens], dtype=np.uint16)
    np.save(cache, arr)
    return arr


def freq_rare_from_counts(counts, vocab, k=100, device="cuda"):
    """Frequent set = top-k token ids; everything else is 'rare'. Also returns the unigram + log-unigram bias init."""
    order = np.argsort(-counts)
    freq_ids = torch.tensor(order[:k].copy(), dtype=torch.long, device=device)
    unigram = torch.tensor(counts / max(counts.sum(), 1), dtype=torch.float32, device=device)
    lu = torch.log(torch.tensor(counts + 1.0, dtype=torch.float32, device=device)); lu -= lu.mean()
    return freq_ids, unigram, lu


def c4_token_stats(tokenizer, vocab, n_docs=2000, max_len=512, device="cuda"):
    """Token counts over a sample of C4 (English, validation) -> used for the marginal + freq/rare token sets.
    Returns (counts ndarray over `vocab`, list of tokenized C4 sequences as LongTensors)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    counts = np.zeros(vocab, dtype=np.int64); seqs = []
    for ex in ds:
        ids = tokenizer(ex["text"], return_tensors="pt").input_ids[0]
        counts += np.bincount(ids.numpy(), minlength=vocab)[:vocab]
        if ids.shape[0] >= max_len:
            seqs.append(ids[:max_len].to(device))
        if len(seqs) >= n_docs:
            break
    return counts, seqs
