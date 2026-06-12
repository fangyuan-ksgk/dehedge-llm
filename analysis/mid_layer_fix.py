"""Fixing mid-layer laziness on TinyStories + GPT.

Diagnosis (see mid_layer_laziness.py): every mid-layer residual has ONE outlier feature dimension (>>100x the
typical magnitude, a self-built constant/bias = massive activation) aligned with an *extreme* singular direction,
while a wide *middle* band of singular directions carries ~zero causal pain. So the model uses neither all its
feature dimensions nor all its singular directions.

Target (math-elegant): a WHITE mid-layer residual, cov = I.
  * off-diagonal of the *correlation* matrix → 0   ⟺  the covariance is diagonal  ⟺  the feature AXES are its
    eigenvectors  ⟺  every feature dimension *is* a singular direction  (the "make dims ↔ dirs the same" ask).
  * the per-dim variances become equal             ⟺  the spectrum is flat  ⟺  no dead directions, no outlier dim
    (every direction equally useful — the "make them all useful / more similar to each other" ask).
Together cov ∝ I is the unique maximally-uniform, permutation-symmetric representation: all dims interchangeable.

The earlier failure ("internal isotropy hurts") used an ABSOLUTE variance floor (force var ≥ 1), which fights the
residual stream's natural growth and its load-bearing constant. Here we test SCALE-FREE whitening (penalize the
*shape* of the covariance, not its scale) so the model keeps its norm budget and only loses the anisotropy.

  python analysis/mid_layer_fix.py --steps 3000 --modes none,decorr,equalize,white,iso
Outputs: results/mid_layer_fix.json , results/figures/mid_layer_fix.png
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _imp(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
dh_gpt = _imp("dh_gpt"); data = _imp("data")
dev = "cuda"; RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures")


def white_terms(z):
    """z: (N,D) centered. Returns (off, eq, vfloor) scale-free + absolute pieces."""
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    var = torch.diagonal(cov)
    std = torch.sqrt(var + 1e-6)
    corr = cov / (std[:, None] * std[None, :])
    off = (corr - torch.diag(torch.diagonal(corr))).pow(2).sum() / (d * (d - 1))   # scale-free decorrelation
    logv = torch.log(var + 1e-6)
    eq = logv.var()                                                                 # scale-free variance equalize
    vfloor = torch.relu(1.0 - std).mean()                                           # absolute floor (the old way)
    return off, eq, vfloor


def reg_loss(reps, mode):
    if mode == "none":
        return torch.zeros((), device=dev)
    tot = torch.zeros((), device=dev)
    for r in reps:
        z = r.reshape(-1, r.size(-1)); z = z - z.mean(0, keepdim=True)
        off, eq, vf = white_terms(z)
        if mode == "decorr":     tot = tot + off
        elif mode == "equalize": tot = tot + eq
        elif mode == "white":    tot = tot + off + eq          # scale-free whitening (the elegant target)
        elif mode == "iso":      tot = tot + off + vf          # original iso_loss (absolute floor) for contrast
    return tot / len(reps)


def get_batch(arr, bs, block, rng):
    ix = rng.integers(0, len(arr) - block - 1, size=bs)
    x = np.stack([arr[i:i + block] for i in ix]); y = np.stack([arr[i + 1:i + 1 + block] for i in ix])
    return torch.tensor(x.astype(np.int64), device=dev), torch.tensor(y.astype(np.int64), device=dev)


def capture_mid(model, mid_idx):
    store = {}; handles = []
    for li in mid_idx:
        handles.append(model.h[li].register_forward_hook(
            lambda m, i, o, li=li: store.__setitem__(li, o)))
    return store, handles


@torch.no_grad()
def evaluate(model, arr, block, rng, freq_ids, iters=60):
    model.eval(); fset = torch.zeros(model.lm_head.weight.shape[0], dtype=torch.bool, device=dev); fset[freq_ids] = True
    ls, lf, lr = [], [], []
    for _ in range(iters):
        x, y = get_batch(arr, 32, block, rng); logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        yf = fset[y.reshape(-1)]; ls.append(ce.mean().item())
        if yf.any(): lf.append(ce[yf].mean().item())
        if (~yf).any(): lr.append(ce[~yf].mean().item())
    model.train()
    return dict(val=float(np.mean(ls)), freqCE=float(np.mean(lf)), rareCE=float(np.mean(lr)))


@torch.no_grad()
def mid_health(model, arr, block, mid_idx):
    """Per mid-block residual: outlier ratio, participation ratio (alive fraction), mean |off-diag corr|."""
    model.eval(); rng = np.random.default_rng(7)
    store, handles = capture_mid(model, mid_idx)
    x, _ = get_batch(arr, 32, block, rng); model(x)
    for h in handles: h.remove()
    out = {}
    for li, r in store.items():
        z = r.reshape(-1, r.size(-1)).float(); z = z - z.mean(0, keepdim=True)
        rms = z.pow(2).mean(0).sqrt()
        cov = (z.T @ z) / (z.shape[0] - 1)
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        pr = float((ev.sum() ** 2 / (ev.pow(2).sum() + 1e-12)) / cov.shape[0])  # participation ratio / D ∈ (0,1]
        std = torch.sqrt(torch.diagonal(cov) + 1e-6)
        corr = cov / (std[:, None] * std[None, :]); d = cov.shape[0]
        offm = float((corr - torch.diag(torch.diagonal(corr))).abs().sum() / (d * (d - 1)))
        out[li] = dict(outlier_ratio=float((rms.max() / rms.median()).item()), part_ratio=pr, off_corr=offm)
    model.train()
    return out


def train_one(mode, coef, arr_tr, arr_va, block, steps, freq_ids, nl, mid_idx, lr=6e-4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = dh_gpt.GPT(50257, 384, nl, 6, block).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    store, handles = capture_mid(model, mid_idx)
    for it in range(steps):
        x, y = get_batch(arr_tr, 32, block, rng)
        logits, loss = model(x, targets=y)
        if mode != "none" and coef > 0:
            loss = loss + coef * reg_loss([store[li] for li in mid_idx], mode)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if it % max(1, steps // 5) == 0:
            print(f"    [{mode}] it={it} loss={loss.item():.3f}", flush=True)
    for h in handles: h.remove()
    q = evaluate(model, arr_va, block, np.random.default_rng(777), freq_ids)
    h = mid_health(model, arr_va, block, mid_idx)
    return dict(quality=q, health=h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--coef", type=float, default=0.05)
    ap.add_argument("--modes", default="none,decorr,equalize,white,iso")
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V)
    freq_ids, _, _ = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    mid_idx = list(range(1, a.nl - 1))   # every block except the first and the last
    print(f"regularizing mid blocks {mid_idx} (of {a.nl}), coef={a.coef}, steps={a.steps}\n", flush=True)
    modes = a.modes.split(","); res = {}
    for mode in modes:
        print(f"#### {mode} ####", flush=True)
        r = train_one(mode, a.coef, tr, va, a.block, a.steps, freq_ids, a.nl, mid_idx)
        res[mode] = r
        hh = r["health"]; pr = np.mean([hh[li]["part_ratio"] for li in hh]); orr = np.mean([hh[li]["outlier_ratio"] for li in hh])
        oc = np.mean([hh[li]["off_corr"] for li in hh])
        print(f"  {mode:9s} val={r['quality']['val']:.3f} freqCE={r['quality']['freqCE']:.3f} rareCE={r['quality']['rareCE']:.3f}"
              f" | aliveFrac={pr:.3f} outlier×={orr:.1f} offCorr={oc:.4f}", flush=True)
        json.dump(res, open(os.path.join(RES, "mid_layer_fix.json"), "w"), indent=2)

    # figure: quality (val/freq/rare) + health (alive fraction, outlier ratio, off-corr) per mode
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(res.keys()); x = np.arange(len(names))
    def hmean(mode, key): hh = res[mode]["health"]; return np.mean([hh[li][key] for li in hh])
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for i, m in enumerate(["val", "freqCE", "rareCE"]):
        ax[0].bar(x + (i - 1) * 0.25, [res[n]["quality"][m] for n in names], 0.25, label=m)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30); ax[0].legend(fontsize=8); ax[0].set_title("quality CE (lower=better)")
    ax[1].bar(x, [hmean(n, "part_ratio") for n in names], color="#2ca02c"); ax[1].axhline(1.0, color="k", ls=":", lw=.7)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30); ax[1].set_title("alive fraction = PR/D (→1 = all dirs useful)")
    ax[2].bar(x, [hmean(n, "outlier_ratio") for n in names], color="#d62728"); ax[2].set_yscale("log")
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=30); ax[2].set_title("outlier dim × median (→1 = no outlier)")
    ax[3].bar(x, [hmean(n, "off_corr") for n in names], color="#9467bd")
    ax[3].set_xticks(x); ax[3].set_xticklabels(names, rotation=30); ax[3].set_title("mean |off-diag corr| (→0 = dims=dirs)")
    plt.suptitle(f"Mid-layer fix on TinyStories — regularize blocks {mid_idx}, coef {a.coef} ({a.steps} steps)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_fix.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_fix.{json,png}", flush=True)


if __name__ == "__main__":
    main()
