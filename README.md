# dehedge-llm
_learning representations without hedging_

A backprop-trained LM is **lazy**: it spends the dominant directions of its readout on the **marginal** (frequent
tokens) and underserves the rare ones, and it parks large **constant** values in feature dimensions its readout barely
reads. This repo (1) **diagnoses** that laziness with singular-direction + causal-ablation analysis on open-source
models (Qwen series, evaluated on C4), and (2) **fixes** it on TinyStories with a marginal **bias channel** + an
**isotropy** regularizer, beating the next-token-prediction baseline (especially on the rare tail).

## Layout
```
isotropy.py                 # iso_loss (the de-hedging reg that worked best) + log_unigram_bias
dh_gpt.py                   # GPT (baseline) and DeHedgeGPT (bias channel + isotropy hooks)
data.py                     # TinyStories token stream + C4 marginal / freq-rare helpers
train_tinystories.py        # compare GPT vs DeHedgeGPT on TinyStories (val / freqCE / rareCE)
analysis/
  last_layer_laziness.py    # singular-direction analysis of the readout W_u
  mid_layer_laziness.py     # per-dimension AND per-singular-direction causal ablation of a mid layer
results/                    # *.json + figures/ produced by the scripts
```

## The fix (TinyStories)
```bash
python train_tinystories.py --steps 8000 --iso 0.10
```
Trains a plain `GPT` and a `DeHedgeGPT` (same architecture/budget), both evaluated with the standard next-token head.
De-hedging = **marginal bias channel** (`isotropy.log_unigram_bias` initializes an additive output bias, so the model
need not encode the unconditional marginal inside the representation) **+ isotropy** (`isotropy.iso_loss` on the final
hidden: a variance-floor that recruits dormant directions + off-diagonal decorrelation). It lowers val CE and, most of
all, **rareCE** — un-starving the tail. Results → `results/train_tinystories.{json,png}`.

## The diagnosis (open-source model + C4)
Both scripts default to `--model Qwen/Qwen3-0.6B` and accept any HF causal-LM id (`--model Qwen/Qwen3-1.7B`, …), using
**C4** to estimate the marginal (token counts) and the frequent/rare token sets.

```bash
python analysis/last_layer_laziness.py --model Qwen/Qwen3-0.6B
python analysis/mid_layer_laziness.py  --model Qwen/Qwen3-0.6B --layer -1   # -1 = middle layer
```

- **last_layer_laziness** — SVD the readout `W_u`; ablate the final hidden state's projection onto chosen singular
  directions and measure freqCE / rareCE / meanCE. Reveals how frequent- vs rare-token signal is distributed across the
  spectrum (largest vs smallest σ) and how ablatable the *middle* band is. Full per-direction results + cumulative
  top-k / bottom-k curves. (Note: the direction of the freq↔rare split is model-dependent — the script reports whatever
  the model actually shows; the robust finding is that a wide middle band is near-free to ablate.)
- **mid_layer_laziness** — at a middle layer: locate the **outlier feature dimension** (often >>100× the typical
  magnitude), report which `W_u` singular direction it aligns with, then **causally ablate every feature dimension and
  every singular direction** (zero / project-out at the mid layer, run the rest of the model) recording
  Δ{mean,freq,rare}CE for each. Demonstrates the outlier's causal dominance vs the negligible average dimension. Full
  per-dim & per-direction arrays in `results/mid_layer_laziness.json`.

## Notes
- `dh_gpt.GPT` is state-dict-compatible with the original `tinystories_gpt.GPT` checkpoints.
- Defaults that worked best: `--iso 0.10`, `--k_freq 100`.
