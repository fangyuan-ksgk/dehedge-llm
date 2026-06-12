"""Causal proof that mid-layer whitening RECRUITS the dead directions.

Diagnosis measured laziness causally: project a direction out of a mid representation, run the rest of the model,
read Delta CE. The dead mid band carried ~zero pain. 'Alive fraction' (covariance participation ratio) is only an
*correlational* proxy for the fix. Here we close the loop with the SAME causal metric: train a baseline GPT and a
whitened GPT (scale-free cov->I on mid blocks), then at a mid block ablate every principal direction of the residual
(project out, run the rest, measure Delta val CE) and compare the per-direction causal-pain spectra.

If whitening worked, the baseline's pain is concentrated in a few top-variance directions (dead tail), while the
whitened model spreads pain across many directions -> a higher CAUSAL participation ratio = more directions actually
used by the rest of the network.

  python analysis/mid_layer_recruit.py --steps 3000 --coef 1.5 --layer 3
Outputs: results/mid_layer_recruit.json , results/figures/mid_layer_recruit.png
"""
import os, sys, json, argparse, numpy as np, torch, torch.nn.functional as F
import importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _imp(p, name):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
mf = _imp(os.path.join(HERE, "analysis", "mid_layer_fix.py"), "mid_layer_fix")
dh_gpt = _imp(os.path.join(HERE, "dh_gpt.py"), "dh_gpt"); data = _imp(os.path.join(HERE, "data.py"), "data")
RES = os.path.join(HERE, "results"); FIG = os.path.join(RES, "figures"); dev = "cuda"


def train(coef, tr, va, block, steps, freq_ids, nl, mid_idx):
    """Train one GPT and return the model (mf.train_one returns only metrics)."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = dh_gpt.GPT(50257, 384, nl, 6, block).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps, pct_start=0.05)
    store, handles = mf.capture_mid(model, mid_idx)
    for it in range(steps):
        x, y = mf.get_batch(tr, 32, block, rng)
        logits, loss = model(x, targets=y)
        if coef > 0:
            loss = loss + coef * mf.reg_loss([store[li] for li in mid_idx], "white")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    for h in handles: h.remove()
    return model


@torch.no_grad()
def causal_pain_spectrum(model, arr, block, L, freq_ids, n_batches=4):
    """At block L: PCA the residual, project out each principal direction, measure mean Delta val CE over batches."""
    model.eval(); rng = np.random.default_rng(11)
    fset = torch.zeros(model.lm_head.weight.shape[0], dtype=torch.bool, device=dev); fset[freq_ids] = True
    batches = [mf.get_batch(arr, 32, block, rng) for _ in range(n_batches)]

    def val_ce(extra_hook):
        h = model.h[L].register_forward_hook(extra_hook) if extra_hook else None
        ces = []
        for x, y in batches:
            logits, _ = model(x)
            ces.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        if h: h.remove()
        return float(np.mean(ces))

    # PCA basis from a fresh capture
    store, hs = mf.capture_mid(model, [L]); x0, _ = mf.get_batch(arr, 32, block, rng); model(x0)
    for h in hs: h.remove()
    z = store[L].reshape(-1, store[L].size(-1)).float(); z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    evals, V = torch.linalg.eigh(cov)               # ascending; columns eigvecs
    order = torch.argsort(evals, descending=True); evals = evals[order]; V = V[:, order]
    base = val_ce(None); D = V.shape[0]; pain = np.zeros(D)
    for i in range(D):
        v = V[:, i]
        def hk(m, inp, out, v=v):
            o = out[0] if isinstance(out, tuple) else out
            o2 = o - (o @ v).unsqueeze(-1) * v
            return ((o2,) + tuple(out[1:])) if isinstance(out, tuple) else o2
        pain[i] = val_ce(hk) - base
    model.train()
    p = np.clip(pain, 0, None); cpr = float((p.sum() ** 2) / (np.sum(p ** 2) + 1e-12)) if p.sum() > 0 else 0.0
    return dict(base=base, pain=pain.tolist(), evals=evals.cpu().numpy().tolist(), causal_part_ratio=cpr,
                var_part_ratio=float((evals.sum() ** 2 / (evals.pow(2).sum() + 1e-12)).item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--block", type=int, default=256); ap.add_argument("--coef", type=float, default=1.5)
    ap.add_argument("--layer", type=int, default=3); a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    arr = data.tinystories_tokens(); V = 50257; sp = int(0.95 * len(arr)); tr, va = arr[:sp], arr[sp:]
    counts = np.bincount(arr.astype(np.int64), minlength=V); freq_ids, _, _ = data.freq_rare_from_counts(counts, V, k=100, device=dev)
    mid_idx = list(range(1, a.nl - 1)); L = a.layer
    out = {}
    for tag, coef in [("baseline", 0.0), ("whitened", a.coef)]:
        print(f"#### training {tag} (coef={coef}) ####", flush=True)
        model = train(coef, tr, va, a.block, a.steps, freq_ids, a.nl, mid_idx)
        sp_ = causal_pain_spectrum(model, va, a.block, L, freq_ids)
        out[tag] = sp_
        print(f"  {tag}: base valCE={sp_['base']:.3f} | causal-PR={sp_['causal_part_ratio']:.1f} var-PR={sp_['var_part_ratio']:.1f}", flush=True)
        json.dump(out, open(os.path.join(RES, "mid_layer_recruit.json"), "w"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for tag, c in [("baseline", "#7f8c8d"), ("whitened", "#2ca02c")]:
        p = np.clip(np.array(out[tag]["pain"]), 0, None)
        ax[0].plot(p, color=c, lw=1.1, label=f"{tag} (causal-PR={out[tag]['causal_part_ratio']:.0f})")
        ev = np.array(out[tag]["evals"]); ax[1].plot(ev / ev.max(), color=c, lw=1.3, label=f"{tag} (var-PR={out[tag]['var_part_ratio']:.0f})")
    ax[0].set_xlabel("principal direction (high→low variance)"); ax[0].set_ylabel("Δ val CE when projected out")
    ax[0].set_title(f"causal pain per mid direction — block {L}"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_yscale("log"); ax[1].set_xlabel("principal direction"); ax[1].set_ylabel("eigenvalue / max")
    ax[1].set_title("mid residual variance spectrum"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle(f"Whitening recruits dead mid directions — TinyStories GPT, coef {a.coef} ({a.steps} steps)", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "mid_layer_recruit.png"), dpi=120, bbox_inches="tight")
    print("\nsaved results/mid_layer_recruit.{json,png}", flush=True)


if __name__ == "__main__":
    main()
