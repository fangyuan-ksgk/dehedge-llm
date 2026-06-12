"""Mid-layer laziness: feature-dimension AND singular-direction causal ablation, for the shared residual AND for
EVERY sub-module's input representation at a mid layer.

Claims tested (real LLM + C4): each mid-layer representation is lazy — there is an OUTLIER feature dimension with
>>100x the typical magnitude; the MIDDLE singular directions carry negligible causal pain (model isn't using them);
only a handful of feature dimensions matter (model isn't using all of them). We give the FULL causal ablation of every
feature dimension and every singular direction (zero / project-out the representation, run the rest of the model,
record Δ{mean,freq,rare}CE) for:
    residual    : the shared mid-layer state, in the readout W_u singular basis  (GLOBAL effect)
    q/k/v/o/gate/up/down : each sub-module's input, in that sub-module's own weight singular basis (LOCAL effect)

  python mid_layer_laziness.py [--model Qwen/Qwen3-0.6B] [--layer -1] [--reps residual,gate_proj,up_proj,...]
Outputs: results/mid_layer_laziness.json (every dim & dir, per representation), results/figures/mid_layer_laziness.png
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); import data as D  # noqa
dev = "cuda"; RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures")
SUBMODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--k_freq", type=int, default=100); ap.add_argument("--n_docs", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=96)
    ap.add_argument("--reps", default="residual," + ",".join(SUBMODS))
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True); reps = a.reps.split(",")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    layers = model.model.layers; NL = len(layers); L = NL // 2 if a.layer < 0 else a.layer
    Wu = model.get_output_embeddings().weight.detach().float(); Vocab = Wu.shape[0]
    counts, seqs = D.c4_token_stats(tok, Vocab, n_docs=max(a.n_docs, 8), max_len=a.maxlen, device=dev)
    freq_ids = torch.tensor(np.argsort(-counts)[:a.k_freq].copy(), device=dev)
    ids = torch.stack([s[:a.maxlen] for s in seqs])[: a.n_docs]
    fset = torch.zeros(Vocab, dtype=torch.bool, device=dev); fset[freq_ids[freq_ids < Vocab]] = True
    y = ids[:, 1:]; lay = layers[L]

    def ce3(logits):
        logp = F.log_softmax(logits[:, :-1].float(), -1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).reshape(-1); yf = fset[y.reshape(-1)]
        return float(ce.mean()), float(ce[yf].mean()), float(ce[~yf].mean())

    def getmod(name):
        return getattr(lay.self_attn, name, None) or getattr(lay.mlp, name, None)

    def capture(name):  # one clean forward, grab the representation that this 'rep' refers to
        store = {}
        if name == "residual":
            h = lay.register_forward_hook(lambda m, i, o: store.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach().float()))
        else:
            h = getmod(name).register_forward_pre_hook(lambda m, i: store.__setitem__("x", i[0].detach().float()))
        model(ids); h.remove(); return store["x"]

    def ablate(name, fn):  # install fn on the right tensor, run rest of model, measure CE
        if name == "residual":
            def hk(m, i, o):
                h0 = o[0] if isinstance(o, tuple) else o
                return ((fn(h0.clone()),) + tuple(o[1:])) if isinstance(o, tuple) else fn(h0.clone())
            H = lay.register_forward_hook(hk)
        else:
            H = getmod(name).register_forward_pre_hook(lambda m, i: (fn(i[0].clone()),) + tuple(i[1:]))
        out = model(ids).logits; H.remove(); return ce3(out)

    base = ce3(model(ids).logits)
    print(f"{a.model}  layer {L}/{NL}  base meanCE={base[0]:.3f} freqCE={base[1]:.3f} rareCE={base[2]:.3f}\n", flush=True)
    results = {"model": a.model, "layer": L, "base": dict(mean=base[0], freq=base[1], rare=base[2]), "reps": {}}
    for name in reps:
        rep = capture(name); Dh = rep.shape[-1]
        W = Wu if name == "residual" else getmod(name).weight.detach().float()
        Vh = torch.linalg.svd(W, full_matrices=False).Vh                       # right-singular vecs (input space)
        rms = rep.reshape(-1, Dh).pow(2).mean(0).sqrt(); c = int(rms.argmax()); mag = float((rms[c] / rms.median()).item())
        i_align = int(Vh[:, c].abs().argmax()); cos = float(Vh[:, c].abs().max()); k = Vh.shape[0]
        dim_m = np.zeros(Dh); dim_f = np.zeros(Dh); dim_r = np.zeros(Dh)
        for j in range(Dh):
            m, f, r = ablate(name, lambda h, j=j: h.index_fill_(-1, torch.tensor([j], device=dev), 0.0))
            dim_m[j], dim_f[j], dim_r[j] = m - base[0], f - base[1], r - base[2]
        sv_m = np.zeros(k); sv_f = np.zeros(k); sv_r = np.zeros(k)
        for i in range(k):
            v = Vh[i]; m, f, r = ablate(name, lambda h, v=v: h - (h @ v).unsqueeze(-1) * v)
            sv_m[i], sv_f[i], sv_r[i] = m - base[0], f - base[1], r - base[2]
        t = k // 3
        bands = {b: float(np.abs(sv_m[s]).mean()) for b, s in [("top", slice(0, t)), ("mid", slice(t, 2 * t)), ("bot", slice(2 * t, k))]}
        results["reps"][name] = dict(D=Dh, outlier_dim=c, mag_ratio=mag, aligned_sv=i_align, sv_rank_pct=100 * i_align / max(k - 1, 1),
                                     align_cos=cos, sv_band_pain=bands, outlier_dim_dmean=float(dim_m[c]),
                                     median_dim_dmean=float(np.median(np.abs(dim_m))),
                                     dim_dmean=dim_m.tolist(), dim_dfreq=dim_f.tolist(), dim_drare=dim_r.tolist(),
                                     sv_dmean=sv_m.tolist(), sv_dfreq=sv_f.tolist(), sv_drare=sv_r.tolist())
        print(f"  {name:10s} D={Dh:4d} outlier dim {c:4d} ({mag:5.0f}x), aligns sv#{i_align} (rank {100*i_align/max(k-1,1):3.0f}%, cos {cos:.2f}) | "
              f"outlier ΔmeanCE={dim_m[c]:+.3f} vs median dim {np.median(np.abs(dim_m)):.4f} ({abs(dim_m[c])/max(np.median(np.abs(dim_m)),1e-6):.0f}x) | "
              f"sv-pain top/mid/bot={bands['top']:.3f}/{bands['mid']:.3f}/{bands['bot']:.3f}", flush=True)
        json.dump(results, open(os.path.join(RES, "mid_layer_laziness.json"), "w"))

    # figure: per-representation summary
    names = list(results["reps"].keys()); x = np.arange(len(names))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].bar(x, [results["reps"][n]["mag_ratio"] for n in names], color="#d62728")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([n.split('_')[0] for n in names], rotation=45, ha="right")
    ax[0].set_title("outlier dim magnitude (×median)"); ax[0].axhline(100, color="k", ls=":", lw=.7)
    for n, c in [("top", "#1f77b4"), ("mid", "#2ca02c"), ("bot", "#ff7f0e")]:
        ax[1].plot(x, [results["reps"][nm]["sv_band_pain"][n] for nm in names], "-o", label=n + " σ-band")
    ax[1].set_xticks(x); ax[1].set_xticklabels([n.split('_')[0] for n in names], rotation=45, ha="right")
    ax[1].set_title("singular-band causal pain (mid = least)"); ax[1].set_ylabel("mean |ΔmeanCE|"); ax[1].legend(fontsize=8)
    ax[2].plot(x, [abs(results["reps"][n]["outlier_dim_dmean"]) for n in names], "-o", color="#d62728", label="outlier dim")
    ax[2].plot(x, [results["reps"][n]["median_dim_dmean"] for n in names], "-o", color="#7f8c8d", label="median dim")
    ax[2].set_yscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels([n.split('_')[0] for n in names], rotation=45, ha="right")
    ax[2].set_title("causal pain: outlier dim vs median dim"); ax[2].legend(fontsize=8)
    plt.suptitle(f"Mid-layer laziness — layer {L} of {a.model}, per representation (C4)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_laziness.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_laziness.{json,png} (full per-dim & per-singular arrays per representation)", flush=True)


if __name__ == "__main__":
    main()
