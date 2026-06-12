"""Mid-layer laziness: feature-dimension AND singular-direction causal ablation of a mid-layer representation.

Claims tested (real LLM + C4), on the mid-layer residual representation:
  * there is an OUTLIER feature dimension with >>100x the magnitude of a typical dimension;
  * despite aligning with the SMALLEST singular directions of the readout W_u (it lives in the readout's near-null
    space), ablating it still causally DOMINATES freqCE, rareCE and meanCE;
  * the AVERAGE feature dimension has negligible causal impact.
We give the full causal-ablation result for EVERY feature dimension and EVERY singular direction of the mid-layer
representation: zero each dim (resp. project out each W_u singular direction) at the mid layer, run the rest of the
model, and record Δ{mean,freq,rare}CE on C4.

  python mid_layer_laziness.py [--model Qwen/Qwen3-0.6B] [--layer -1=mid] [--n_docs 4 --maxlen 128]
Outputs: results/mid_layer_laziness.json (every dim & dir), results/figures/mid_layer_laziness.png
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
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--k_freq", type=int, default=100); ap.add_argument("--n_docs", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=128); a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    layers = model.model.layers; NL = len(layers); L = NL // 2 if a.layer < 0 else a.layer
    Wu = model.get_output_embeddings().weight.detach().float(); Vocab, Dh = Wu.shape
    counts, seqs = D.c4_token_stats(tok, Vocab, n_docs=max(a.n_docs, 8), max_len=a.maxlen, device=dev)
    freq_ids = torch.tensor(np.argsort(-counts)[:a.k_freq].copy(), device=dev)
    ids = torch.stack([s[:a.maxlen] for s in seqs])[: a.n_docs]
    fset = torch.zeros(Vocab, dtype=torch.bool, device=dev); fset[freq_ids[freq_ids < Vocab]] = True
    y = ids[:, 1:]

    def ce3(logits):
        logp = F.log_softmax(logits[:, :-1].float(), -1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).reshape(-1); yf = fset[y.reshape(-1)]
        return float(ce.mean()), float(ce[yf].mean()), float(ce[~yf].mean())

    # ---- characterize the mid-layer residual ----
    hL = model(ids, output_hidden_states=True).hidden_states[L].float()      # (B,T,D)
    rms = hL.reshape(-1, Dh).pow(2).mean(0).sqrt(); c_out = int(rms.argmax())
    mag_ratio = float((rms[c_out] / rms.median()).item())
    U, S, Vh = torch.linalg.svd(Wu, full_matrices=False)                     # Vh (D,D) right-sing vecs (W_u readout)
    cos_c = Vh[:, c_out].abs(); i_align = int(cos_c.argmax()); align = float(cos_c.max())
    print(f"{a.model}  layer {L}/{NL}  D={Dh}", flush=True)
    print(f"  outlier feature dim = {c_out}, magnitude = {mag_ratio:.0f}x median channel", flush=True)
    print(f"  aligns with W_u singular dir #{i_align}/{Dh} (sigma-rank {100*i_align/(Dh-1):.0f}% : "
          f"{'BOTTOM/null' if i_align > 0.9*Dh else 'top' if i_align < 0.1*Dh else 'mid'}), cosine {align:.2f}", flush=True)

    # ---- causal ablation: hook the mid layer output, run rest of model, measure CE ----
    def run_ablated(fn):
        def hook(mod, inp, out):
            h0 = out[0] if isinstance(out, tuple) else out
            return ((fn(h0.clone()),) + tuple(out[1:])) if isinstance(out, tuple) else fn(h0.clone())
        H = layers[L].register_forward_hook(hook); logits = model(ids).logits; H.remove(); return ce3(logits)

    base = ce3(model(ids).logits); print(f"  base meanCE={base[0]:.3f} freqCE={base[1]:.3f} rareCE={base[2]:.3f}", flush=True)
    # every FEATURE DIMENSION
    dim_dmean = np.zeros(Dh); dim_dfreq = np.zeros(Dh); dim_drare = np.zeros(Dh)
    for c in range(Dh):
        m, f, r = run_ablated(lambda h, c=c: h.index_fill_(-1, torch.tensor([c], device=dev), 0.0))
        dim_dmean[c] = m - base[0]; dim_dfreq[c] = f - base[1]; dim_drare[c] = r - base[2]
        if c % 256 == 0: print(f"    dim {c}/{Dh} ...", flush=True)
    # every SINGULAR DIRECTION (of W_u, projected out of the mid residual)
    sv_dmean = np.zeros(Dh); sv_dfreq = np.zeros(Dh); sv_drare = np.zeros(Dh)
    for i in range(Dh):
        v = Vh[i]
        m, f, r = run_ablated(lambda h, v=v: h - (h @ v).unsqueeze(-1) * v)
        sv_dmean[i] = m - base[0]; sv_dfreq[i] = f - base[1]; sv_drare[i] = r - base[2]
        if i % 256 == 0: print(f"    sv {i}/{Dh} ...", flush=True)

    out = dict(model=a.model, layer=L, D=Dh, outlier_dim=c_out, mag_ratio=mag_ratio,
               outlier_aligned_sv=i_align, outlier_align_cos=align, sigma_rank_pct=100*i_align/(Dh-1),
               base=dict(mean=base[0], freq=base[1], rare=base[2]),
               dim_dmean=dim_dmean.tolist(), dim_dfreq=dim_dfreq.tolist(), dim_drare=dim_drare.tolist(),
               sv_dmean=sv_dmean.tolist(), sv_dfreq=sv_dfreq.tolist(), sv_drare=sv_drare.tolist())
    json.dump(out, open(os.path.join(RES, "mid_layer_laziness.json"), "w"))
    # summary
    avg_dim = float(np.median(np.abs(dim_dmean))); out_dim = float(abs(dim_dmean[c_out]))
    print(f"\n  ablate OUTLIER dim {c_out}: ΔmeanCE={dim_dmean[c_out]:+.3f} ΔfreqCE={dim_dfreq[c_out]:+.3f} ΔrareCE={dim_drare[c_out]:+.3f}", flush=True)
    print(f"  median |ΔmeanCE| over all dims = {avg_dim:.4f}  -> outlier is {out_dim/max(avg_dim,1e-6):.0f}x the typical dim's impact", flush=True)
    # figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 9)); idx = np.arange(Dh); t = Dh // 3
    rmsn = rms.cpu().numpy()
    ax[0,0].plot(rmsn, lw=.6); ax[0,0].plot([c_out], [rmsn[c_out]], "ro"); ax[0,0].set_yscale("log")
    ax[0,0].set_title(f"feature-dim magnitude (rms) — outlier dim {c_out} = {mag_ratio:.0f}x median"); ax[0,0].set_xlabel("feature dim")
    ax[0,1].plot(cos_c.cpu().numpy(), color="#8e44ad", lw=.6); ax[0,1].plot([i_align],[align],"ro")
    ax[0,1].set_title(f"cos(outlier dim, W_u singular dirs) — peaks at #{i_align} (σ-rank {100*i_align/(Dh-1):.0f}%)"); ax[0,1].set_xlabel("singular dir i")
    ax[1,0].plot(dim_dmean, lw=.6, label="ΔmeanCE"); ax[1,0].plot(dim_dfreq, lw=.6, label="ΔfreqCE"); ax[1,0].plot(dim_drare, lw=.6, label="ΔrareCE")
    ax[1,0].plot([c_out],[dim_dmean[c_out]],"ro"); ax[1,0].set_title("causal ablation per FEATURE DIM (outlier dominates; avg≈0)"); ax[1,0].set_xlabel("feature dim"); ax[1,0].legend(fontsize=8)
    ax[1,1].plot(sv_dmean, lw=.6, label="ΔmeanCE"); ax[1,1].plot(sv_dfreq, lw=.6, label="ΔfreqCE"); ax[1,1].plot(sv_drare, lw=.6, label="ΔrareCE")
    ax[1,1].axvspan(t, 2*t, color="orange", alpha=.12); ax[1,1].set_title("causal ablation per SINGULAR DIR (of W_u)"); ax[1,1].set_xlabel("singular dir i (large→small σ)"); ax[1,1].legend(fontsize=8)
    plt.suptitle(f"Mid-layer laziness — layer {L} of {a.model} (C4)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_laziness.png"), dpi=120, bbox_inches="tight")
    print("saved results/mid_layer_laziness.{json,png}", flush=True)


if __name__ == "__main__":
    main()
