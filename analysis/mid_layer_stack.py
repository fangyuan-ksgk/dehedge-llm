"""Do the two de-hedging fixes STACK? Last-layer (bias channel + final isotropy) vs mid-layer (scale-free whitening).

We have two independent fixes for two kinds of laziness:
  * LAST layer  : marginal bias channel + iso_loss on the final hidden state (un-starves the rare tail at the readout).
  * MID  layers : white_loss (scale-free cov->I) on mid block residuals (recruits the dead mid direction band).
This trains all four cells of the 2x2 (none / last / mid / both) at equal budget and compares val/freq/rareCE, to see
whether the mid-layer fix adds on top of the last-layer fix or conflicts with it.

  python analysis/mid_layer_stack.py --steps 6000 --mid_coef 0.6 --last_iso 0.10
Outputs: results/mid_layer_stack.json , results/figures/mid_layer_stack.png
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _imp(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
dh_gpt = _imp(os.path.join(HERE, "dh_gpt.py"), "dh_gpt"); isotropy = _imp(os.path.join(HERE, "isotropy.py"), "isotropy")
data = _imp(os.path.join(HERE, "data.py"), "data"); mf = _imp(os.path.join(HERE, "analysis", "mid_layer_fix.py"), "mid_layer_fix")
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures"); dev = "cuda"


@torch.no_grad()
def evaluate(model, arr, block, freq_ids, iters=80):
    model.eval(); rng = np.random.default_rng(777)
    fset = torch.zeros(model.lm_head.weight.shape[0], dtype=torch.bool, device=dev); fset[freq_ids] = True
    ls, lf, lr = [], [], []
    for _ in range(iters):
        x, y = mf.get_batch(arr, 32, block, rng); logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        yf = fset[y.reshape(-1)]; ls.append(ce.mean().item())
        if yf.any(): lf.append(ce[yf].mean().item())
        if (~yf).any(): lr.append(ce[~yf].mean().item())
    model.train()
    return dict(val=float(np.mean(ls)), freqCE=float(np.mean(lf)), rareCE=float(np.mean(lr)))


def train_cell(use_last, mid_coef, tr, va, block, steps, freq_ids, nl, mid_idx, lu):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = dh_gpt.GPT(50257, 384, nl, 6, block, bias_channel=use_last, ubias_init=(lu if use_last else None)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps, pct_start=0.05)
    store, handles = mf.capture_mid(model, mid_idx)
    for it in range(steps):
        x, y = mf.get_batch(tr, 32, block, rng)
        logits, loss, rep = model(x, targets=y, return_rep=True)
        if use_last:
            loss = loss + 0.10 * isotropy.iso_loss(rep)
        if mid_coef > 0:
            loss = loss + mid_coef * torch.stack([isotropy.white_loss(store[li]) for li in mid_idx]).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if it % max(1, steps // 4) == 0:
            print(f"      it={it} loss={loss.item():.3f}", flush=True)
    for h in handles: h.remove()
    return evaluate(model, va, block, freq_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--mid_coef", type=float, default=0.6)
    ap.add_argument("--last_iso", type=float, default=0.10); a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V)
    freq_ids, _, lu = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    mid_idx = list(range(1, a.nl - 1))
    cells = {"none": (False, 0.0), "last (bias+iso)": (True, 0.0),
             "mid (whiten)": (False, a.mid_coef), "both": (True, a.mid_coef)}
    res = {}
    for name, (use_last, mc) in cells.items():
        print(f"#### {name} ####", flush=True)
        res[name] = train_cell(use_last, mc, tr, va, a.block, a.steps, freq_ids, a.nl, mid_idx, lu)
        print(f"  {name:18s} val={res[name]['val']:.3f} freqCE={res[name]['freqCE']:.3f} rareCE={res[name]['rareCE']:.3f}", flush=True)
        json.dump(res, open(os.path.join(RES, "mid_layer_stack.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(res.keys()); mets = ["val", "freqCE", "rareCE"]; x = np.arange(len(names)); w = 0.25
    colors = {"val": "#1f77b4", "freqCE": "#ff7f0e", "rareCE": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(mets):
        vals = [res[n][m] for n in names]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=colors[m])
        for j, v in enumerate(vals): ax.text(x[j] + (i - 1) * w, v + .02, f"{v:.2f}", ha="center", fontsize=7)
    base = res["none"]
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.legend()
    ax.set_title(f"De-hedging stacks? last-layer + mid-layer fixes ({a.steps} steps)\n"
                 f"both vs none: Δval {res['both']['val']-base['val']:+.3f}  ΔrareCE {res['both']['rareCE']-base['rareCE']:+.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_stack.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_stack.{json,png}", flush=True)


if __name__ == "__main__":
    main()
