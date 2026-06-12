"""Last-layer laziness: singular-direction analysis of the readout W_u (unembedding).

Claim tested: the readout representation is 'lazy' — the *biggest* singular directions carry the frequent-token
signal (freqCE), the *smallest* carry the rare-token signal (rareCE), and a wide *middle* band can be causally
ablated from the final hidden state with little effect on either.

Method (real LLM + C4): SVD W_u = U S Vᵀ. Take the model's final hidden state h on C4 text; ablate the projection of
h onto chosen right-singular directions (h ← h − (h·v)v), recompute logits = h'·W_uᵀ, and measure freqCE / rareCE /
meanCE. We sweep (a) cumulative TOP-k and BOTTOM-k, and (b) every single singular direction.

  python last_layer_laziness.py [--model Qwen/Qwen3-0.6B] [--k_freq 100] [--n_docs 24]
Outputs: results/last_layer_laziness.json , results/figures/last_layer_laziness.png
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); import data as D  # noqa
dev = "cuda"; RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--k_freq", type=int, default=100)
    ap.add_argument("--n_docs", type=int, default=24); ap.add_argument("--maxlen", type=int, default=256)
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    V = model.config.hidden_size  # placeholder; real V below
    Wu = model.get_output_embeddings().weight.detach().float()      # (Vocab, D)
    Vocab, Dh = Wu.shape
    counts, seqs = D.c4_token_stats(tok, Vocab, n_docs=a.n_docs, max_len=a.maxlen, device=dev)
    freq_ids = torch.tensor(np.argsort(-counts)[:a.k_freq].copy(), device=dev)
    ids = torch.stack([s[:a.maxlen] for s in seqs])[: a.n_docs]
    U, S, Vh = torch.linalg.svd(Wu, full_matrices=False)             # Vh (D,D) right-singular vecs (rows)
    h = model(ids, output_hidden_states=True).hidden_states[-1].float()  # (B,T,D) final hidden (post-norm)
    y = ids[:, 1:]; hp = h[:, :-1]                                   # predict next token
    fset = torch.zeros(Vocab, dtype=torch.bool, device=dev); fset[freq_ids[freq_ids < Vocab]] = True
    base_logits = hp @ Wu.T

    def split_ce(logits):
        logp = F.log_softmax(logits, -1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).reshape(-1)
        yf = fset[y.reshape(-1)]
        return float(ce.mean()), float(ce[yf].mean()), float(ce[~yf].mean())  # mean, freq, rare

    base = split_ce(base_logits)
    print(f"{a.model}  D={Dh} Vocab={Vocab}  base meanCE={base[0]:.3f} freqCE={base[1]:.3f} rareCE={base[2]:.3f}", flush=True)
    WuV = (Wu @ Vh.T)                                                # (Vocab, D): columns = W_u v_i
    hpV = (hp @ Vh.T)                                                # (B,T,D): coeff h·v_i

    def ablate_set(idxs):  # remove directions idxs from hp (rank update on logits)
        d = base_logits - torch.einsum('btk,vk->btv', hpV[..., idxs], WuV[:, idxs])
        return split_ce(d)

    # (a) cumulative top-k / bottom-k
    ks = sorted(set([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, Dh // 2, Dh - 1]))
    ks = [k for k in ks if 1 <= k < Dh]
    topf, topr, botf, botr = [], [], [], []
    for k in ks:
        _, f, r = ablate_set(torch.arange(0, k, device=dev)); topf.append(f - base[1]); topr.append(r - base[2])
        _, f, r = ablate_set(torch.arange(Dh - k, Dh, device=dev)); botf.append(f - base[1]); botr.append(r - base[2])
    # (b) per-single-direction
    df = np.zeros(Dh); dr = np.zeros(Dh)
    for i in range(Dh):
        _, f, r = ablate_set(torch.tensor([i], device=dev)); df[i] = f - base[1]; dr[i] = r - base[2]
    # band summary (thirds)
    t = Dh // 3
    bands = {b: dict(dFreq=float(df[s].mean()), dRare=float(dr[s].mean())) for b, s in
             [("top", slice(0, t)), ("mid", slice(t, 2 * t)), ("bottom", slice(2 * t, Dh))]}
    print("band-mean single-direction ablation impact:")
    for b, v in bands.items():
        print(f"  {b:7s}: ΔfreqCE={v['dFreq']:+.4f}  ΔrareCE={v['dRare']:+.4f}", flush=True)
    json.dump(dict(model=a.model, base=dict(mean=base[0], freq=base[1], rare=base[2]),
                   ks=ks, top_dfreq=topf, top_drare=topr, bot_dfreq=botf, bot_drare=botr,
                   per_dir_dfreq=df.tolist(), per_dir_drare=dr.tolist(), bands=bands),
              open(os.path.join(RES, "last_layer_laziness.json"), "w"))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ks, topf, "-o", color="#1f77b4", label="ablate TOP-k → ΔfreqCE")
    ax[0].plot(ks, topr, "--o", color="#1f77b4", label="ablate TOP-k → ΔrareCE")
    ax[0].plot(ks, botf, "-s", color="#d62728", label="ablate BOTTOM-k → ΔfreqCE")
    ax[0].plot(ks, botr, "--s", color="#d62728", label="ablate BOTTOM-k → ΔrareCE")
    ax[0].set_xscale("log"); ax[0].set_xlabel("k singular dirs removed"); ax[0].set_ylabel("Δ CE"); ax[0].legend(fontsize=8)
    ax[0].set_title("cumulative top vs bottom ablation"); ax[0].grid(alpha=.3)
    ax[1].plot(df, color="#1f77b4", lw=.7, label="ΔfreqCE"); ax[1].plot(dr, color="#d62728", lw=.7, label="ΔrareCE")
    ax[1].axvspan(t, 2 * t, color="orange", alpha=.12, label="mid band")
    ax[1].set_xlabel("singular direction i (large→small σ)"); ax[1].set_ylabel("Δ CE (single-dir ablation)")
    ax[1].legend(fontsize=8); ax[1].set_title("per-direction ablation impact"); ax[1].grid(alpha=.3)
    plt.suptitle(f"Last-layer laziness — singular structure of W_u ({a.model}, C4)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "last_layer_laziness.png"), dpi=120, bbox_inches="tight")
    print("saved results/last_layer_laziness.{json,png}", flush=True)


if __name__ == "__main__":
    main()
