"""Does causal recruitment saturate, or track variance? (the task-bounded vs optimization-bounded test)

mid_layer_recruit.py found whitening lifts the mid-residual VARIANCE participation ratio ~9x but the CAUSAL
participation ratio (directions the rest of the net actually reads) only ~2x. This sweeps whitening strength and
plots both PRs vs coef. If causal-PR plateaus while var-PR keeps climbing -> filling directions with energy can't
manufacture causal need; causal usefulness is capped by task dimension (a clean law). If they track -> push harder.

  python analysis/mid_layer_recruit_sweep.py --steps 3000 --coefs 0,0.2,0.6,1.5,4.0,10.0 --layer 3
Outputs: results/mid_layer_recruit_sweep.json , results/figures/mid_layer_recruit_sweep.png
"""
import os, sys, json, argparse, numpy as np, torch
import importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _imp(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
rc = _imp(os.path.join(HERE, "analysis", "mid_layer_recruit.py"), "mid_layer_recruit")
data = _imp(os.path.join(HERE, "data.py"), "data")
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures"); dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--coefs", default="0,0.2,0.6,1.5,4.0,10.0")
    ap.add_argument("--layer", type=int, default=3); a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V); freq_ids, _, _ = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    mid_idx = list(range(1, a.nl - 1)); coefs = [float(c) for c in a.coefs.split(",")]
    res = {}
    for c in coefs:
        print(f"#### coef={c} ####", flush=True)
        model = rc.train(c, tr, va, a.block, a.steps, freq_ids, a.nl, mid_idx)
        s = rc.causal_pain_spectrum(model, va, a.block, a.layer, freq_ids)
        res[str(c)] = dict(val=s["base"], causal_pr=s["causal_part_ratio"], var_pr=s["var_part_ratio"])
        print(f"  coef={c}: val={s['base']:.3f} causal-PR={s['causal_part_ratio']:.1f} var-PR={s['var_part_ratio']:.1f}", flush=True)
        json.dump(res, open(os.path.join(RES, "mid_layer_recruit_sweep.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    xt = [str(c) for c in coefs]
    cpr = [res[str(c)]["causal_pr"] for c in coefs]; vpr = [res[str(c)]["var_pr"] for c in coefs]; val = [res[str(c)]["val"] for c in coefs]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(xt, vpr, "-o", color="#2ca02c", label="variance-PR (energy)")
    ax[0].plot(xt, cpr, "-s", color="#d62728", label="causal-PR (network reads)")
    ax[0].set_xlabel("whitening coef"); ax[0].set_ylabel("participation ratio (# directions)")
    ax[0].set_title("variance vs causal recruitment"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax2 = ax[1]; ax2.plot(xt, val, "-o", color="#1f77b4", label="val CE"); ax2.set_xlabel("whitening coef")
    ax2.set_ylabel("val CE"); ax2.set_title("quality vs strength"); ax2.legend(); ax2.grid(alpha=.3)
    plt.suptitle(f"Causal vs variance recruitment under whitening — TinyStories GPT block {a.layer} ({a.steps} steps)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_recruit_sweep.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_recruit_sweep.{json,png}", flush=True)


if __name__ == "__main__":
    main()
