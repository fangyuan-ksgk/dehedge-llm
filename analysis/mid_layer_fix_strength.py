"""How far can scale-free whitening push the mid residual toward cov = I?

The sweep (mid_layer_fix.py) found `white` (off-diag corr + log-var equalize) wins on both quality and health, but
at coef 0.05 the alive fraction (PR/D) is only ~0.05 — the knob is right but barely turned. This script traces the
ALIVE-FRACTION ↔ QUALITY frontier as we increase the whitening strength, to see whether driving cov -> I harder keeps
recruiting dead directions / dissolving the outlier, and what (if anything) it costs in CE.

  python analysis/mid_layer_fix_strength.py --steps 3000 --coefs 0,0.05,0.2,0.6,1.5,4.0
Outputs: results/mid_layer_fix_strength.json , results/figures/mid_layer_fix_strength.png
"""
import os, sys, json, argparse, numpy as np, torch
import importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _imp(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mf = _imp(os.path.join(HERE, "analysis", "mid_layer_fix.py"), "mid_layer_fix")
data = _imp(os.path.join(HERE, "data.py"), "data")
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures"); dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--coefs", default="0,0.05,0.2,0.6,1.5,4.0")
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V)
    freq_ids, _, _ = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    mid_idx = list(range(1, a.nl - 1)); coefs = [float(c) for c in a.coefs.split(",")]
    print(f"whitening-strength sweep on mid blocks {mid_idx}, coefs={coefs}\n", flush=True)
    res = {}
    for c in coefs:
        mode = "none" if c == 0 else "white"
        print(f"#### white coef={c} ####", flush=True)
        r = mf.train_one(mode, c, tr, va, a.block, a.steps, freq_ids, a.nl, mid_idx)
        hh = r["health"]
        r["alive"] = float(np.mean([hh[li]["part_ratio"] for li in hh]))
        r["outlier"] = float(np.mean([hh[li]["outlier_ratio"] for li in hh]))
        r["offcorr"] = float(np.mean([hh[li]["off_corr"] for li in hh]))
        res[str(c)] = r
        print(f"  coef={c}: val={r['quality']['val']:.3f} rareCE={r['quality']['rareCE']:.3f} | "
              f"alive={r['alive']:.3f} outlier×={r['outlier']:.1f} offCorr={r['offcorr']:.4f}", flush=True)
        json.dump(res, open(os.path.join(RES, "mid_layer_fix_strength.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cs = coefs; val = [res[str(c)]["quality"]["val"] for c in cs]; rare = [res[str(c)]["quality"]["rareCE"] for c in cs]
    alive = [res[str(c)]["alive"] for c in cs]; outl = [res[str(c)]["outlier"] for c in cs]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    xt = [str(c) for c in cs]
    ax[0].plot(xt, val, "-o", color="#1f77b4", label="val CE"); ax[0].plot(xt, rare, "-s", color="#2ca02c", label="rareCE")
    ax[0].set_xlabel("whitening coef"); ax[0].set_title("quality vs whitening strength"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(xt, alive, "-o", color="#2ca02c"); ax[1].axhline(1.0, color="k", ls=":", lw=.7)
    ax[1].set_xlabel("whitening coef"); ax[1].set_title("alive fraction PR/D (→1 = all dirs useful)"); ax[1].grid(alpha=.3)
    ax[2].plot(xt, outl, "-o", color="#d62728"); ax[2].axhline(1.0, color="k", ls=":", lw=.7)
    ax[2].set_xlabel("whitening coef"); ax[2].set_yscale("log"); ax[2].set_title("outlier dim × median (→1 = gone)"); ax[2].grid(alpha=.3)
    plt.suptitle(f"Mid-layer whitening strength frontier — TinyStories GPT, blocks {mid_idx} ({a.steps} steps)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_fix_strength.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_fix_strength.{json,png}", flush=True)


if __name__ == "__main__":
    main()
