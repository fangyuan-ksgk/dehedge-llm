"""Train and compare a plain GPT (next-token baseline) vs DeHedgeGPT (bias channel + isotropy) on TinyStories.

Both share architecture, data, and budget; both are evaluated with the standard next-token head, split into
frequent / rare token CE. De-hedging is expected to beat the baseline overall and especially on the rare tail.

  python train_tinystories.py [--steps 8000] [--ne 384 --nl 6 --nh 6] [--iso 0.10]
Outputs: results/train_tinystories.json , results/figures/train_tinystories.png , checkpoints in results/.
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
def _imp(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
dh_gpt = _imp("dh_gpt"); isotropy = _imp("isotropy"); data = _imp("data")
dev = "cuda"
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures")


def get_batch(arr, bs, block, rng):
    ix = rng.integers(0, len(arr) - block - 1, size=bs)
    x = np.stack([arr[i:i + block] for i in ix]); y = np.stack([arr[i + 1:i + 1 + block] for i in ix])
    return (torch.tensor(x.astype(np.int64), device=dev), torch.tensor(y.astype(np.int64), device=dev))


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
    return dict(val=float(np.mean(ls)), ppl=float(np.exp(np.mean(ls))), freqCE=float(np.mean(lf)), rareCE=float(np.mean(lr)))


def train_one(model, iso, arr_tr, arr_va, block, steps, freq_ids, lr=6e-4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    for it in range(steps):
        x, y = get_batch(arr_tr, 32, block, rng)
        logits, loss, rep = model(x, targets=y, return_rep=True)
        if iso > 0:
            loss = loss + iso * isotropy.iso_loss(rep)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if it % max(1, steps // 6) == 0:
            ev = evaluate(model, arr_va, block, np.random.default_rng(123), freq_ids, iters=20)
            print(f"    it={it} val={ev['val']:.3f} rareCE={ev['rareCE']:.3f}", flush=True)
    return evaluate(model, arr_va, block, np.random.default_rng(777), freq_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--ne", type=int, default=384)
    ap.add_argument("--nl", type=int, default=6); ap.add_argument("--nh", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--iso", type=float, default=0.10)
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V)
    freq_ids, _, lu = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    res = {}
    print("#### GPT (next-token baseline) ####", flush=True)
    res["GPT"] = train_one(dh_gpt.GPT(V, a.ne, a.nl, a.nh, a.block), 0.0, tr, va, a.block, a.steps, freq_ids)
    print(f"  GPT: val={res['GPT']['val']:.3f} rareCE={res['GPT']['rareCE']:.3f}", flush=True)
    print("#### DeHedgeGPT (bias channel + isotropy) ####", flush=True)
    res["DeHedgeGPT"] = train_one(dh_gpt.DeHedgeGPT(V, ubias_init=lu, ne=a.ne, nl=a.nl, nh=a.nh, block=a.block),
                                  a.iso, tr, va, a.block, a.steps, freq_ids)
    print(f"  DeHedgeGPT: val={res['DeHedgeGPT']['val']:.3f} rareCE={res['DeHedgeGPT']['rareCE']:.3f}", flush=True)
    res["delta"] = {k: res["DeHedgeGPT"][k] - res["GPT"][k] for k in ("val", "freqCE", "rareCE")}
    json.dump(res, open(os.path.join(RES, "train_tinystories.json"), "w"), indent=2)
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mets = ["val", "freqCE", "rareCE"]; x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(x - w/2, [res["GPT"][m] for m in mets], w, label="GPT (baseline)", color="#7f8c8d")
    ax.bar(x + w/2, [res["DeHedgeGPT"][m] for m in mets], w, label="DeHedgeGPT", color="#2ca02c")
    for i, m in enumerate(mets):
        ax.text(i - w/2, res["GPT"][m] + .02, f"{res['GPT'][m]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, res["DeHedgeGPT"][m] + .02, f"{res['DeHedgeGPT'][m]:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["val CE", "freqCE", "rareCE"]); ax.legend()
    ax.set_title(f"TinyStories: GPT vs DeHedgeGPT (Δval {res['delta']['val']:+.3f}, ΔrareCE {res['delta']['rareCE']:+.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "train_tinystories.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/train_tinystories.json + figure", flush=True)


if __name__ == "__main__":
    main()
