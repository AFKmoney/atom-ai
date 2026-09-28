"""Field-saturation diagnostic (eval/no_grad, single-threaded).

For each checkpoint (optionally with an eval-time --field-leak override, i.e. a
frozen-weight simulation of the lever):
  A. one long episode from reset over contiguous corpus text (atoms flushed every
     64 ticks like training): first tick at the field_max_rms cap, fraction of
     ticks at cap, RMS trajectory, energy share of the linearly unstable modes,
     temporal cosine of α(t) vs α(t+k).
  B. shared-prefix test: after a prefix of P ticks, fork the full model state and
     feed 6 different 64-byte continuations (3 chat prompts + 3 corpus snippets);
     mean pairwise cosine of the resulting α (flattened) and of a_hat (the
     normalized 64-d JL features the α-MLP actually reads). ~1.0 = the recent
     text does not change the field.
  C. linear decodability: softmax regression next-byte (and current-byte) from
     a_hat collected in A (train first 70%, test last 30%) vs unigram CE.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chat_atom_native import load_model  # noqa: E402
from src.io.atomizer import Atomizer  # noqa: E402

PROMPTS = ["Bonjour, comment ça va ?", "Qui es-tu ?", "Il était une fois"]


def packets_of(text: str):
    return Atomizer(max_span_bytes=1).encode(text, reset=True)


def a_hat_of(model, alpha):
    a = model.surface.alpha_only_features(alpha)
    return a / a.pow(2).mean().sqrt().clamp_min(1e-8)


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def mean_pairwise_cos(vs):
    vals = [cos(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
    return sum(vals) / len(vals)


@torch.no_grad()
def tick(model, pkt, t, flush_every=64):
    out = model.forward_packet(pkt)
    if flush_every and t % flush_every == 0:
        model.core.atoms = type(model.core.atoms)()
        model.core.energy_history = []
    return out


@torch.no_grad()
def episode(model, pkts, snap_at=()):
    model.reset_state(reset_atomizer=True)
    model.eval()
    scales, rms, ahat, alphas, snaps = [], [], [], [], {}
    for t, p in enumerate(pkts, start=1):
        out = tick(model, p, t)
        scales.append(out["field_scale"])
        rms.append(out["field_rms_before"])
        alphas.append(out["field"].detach().reshape(-1).clone())
        ahat.append(a_hat_of(model, out["field"]).detach().clone())
        if t in snap_at:
            snaps[t] = copy.deepcopy(model)
    return scales, rms, alphas, ahat, snaps


def mode_energy(model, alpha_flat):
    nm, d = model.core.state.n_modes, model.core.state.d_model
    e = alpha_flat.reshape(nm, d).pow(2).sum(dim=1)
    share = e / e.sum().clamp_min(1e-12)
    ph = torch.arange(nm, dtype=torch.float32) * (2 * math.pi / nm)
    ed = float(model.core.dynamics.dynamics.energy_decay.item())
    unstable = torch.sin(ph) > (1.0 - ed)  # linear rate sin(φ_m)+(ed-1) > 0
    top = torch.topk(share, 3)
    return {
        "unstable_modes": [int(i) for i in torch.nonzero(unstable).flatten()],
        "energy_share_unstable_modes": float(share[unstable].sum()),
        "top3_modes": [int(i) for i in top.indices],
        "top3_share": [round(float(v), 4) for v in top.values],
    }


def softmax_probe(X, y, n_classes=256, wd=1e-3, iters=200):
    n = X.shape[0]
    ntr = int(0.7 * n)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xtr @ W + b, ytr) + wd * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        te = float(F.cross_entropy(Xte @ W + b, yte))
        counts = torch.bincount(ytr, minlength=n_classes).float() + 0.5
        uni = float(F.cross_entropy(torch.log(counts / counts.sum()).expand(len(yte), -1), yte))
    return {"test_ce": te, "unigram_ce": uni, "n_train": ntr, "n_test": n - ntr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="tag=path")
    ap.add_argument("--leaks", type=float, nargs="+", default=[None], help="eval-time field_leak override(s); omit = ckpt value")
    ap.add_argument("--episode-ticks", type=int, default=8000)
    ap.add_argument("--offset", type=int, default=123457)
    ap.add_argument("--prefixes", type=int, nargs="+", default=[256, 3000])
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--probe", action="store_true", help="run section C (softmax probes)")
    ap.add_argument("--data", default="data/corpus_fr_medium.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    raw = Path(args.data).read_bytes()
    text = raw[args.offset: args.offset + args.episode_ticks + 1].decode("utf-8", errors="ignore")
    pkts = packets_of(text)[: args.episode_ticks + 1]
    conts = [p for p in PROMPTS]
    for off in (37000, 211000, 402000):
        conts.append(raw[off: off + args.cont_len].decode("utf-8", errors="ignore"))
    conts = [packets_of(c)[: args.cont_len] for c in conts]
    results = {"episode_ticks": len(pkts) - 1, "offset": args.offset, "rows": []}
    for spec in args.checkpoints:
        tag, path = spec.split("=", 1)
        base = load_model(path)
        for leak in args.leaks:
            t0 = time.time()
            model = copy.deepcopy(base)
            if leak is not None:
                model.field_leak = float(leak)
            dyn = model.core.dynamics.dynamics
            row = {
                "tag": tag, "checkpoint": path, "field_leak": float(model.field_leak),
                "field_max_rms": model.field_max_rms,
                "energy_decay": float(dyn.energy_decay), "coupling_scale": float(dyn.coupling_scale),
                "phase_sync": float(dyn.phase_sync), "dt": float(model.core.dynamics.dt),
            }
            scales, rms, alphas, ahat, snaps = episode(model, pkts[:-1], snap_at=set(args.prefixes))
            at_cap = [s < 0.999999 for s in scales]
            n = len(at_cap)
            row["first_cap_tick"] = next((i + 1 for i, c in enumerate(at_cap) if c), None)
            row["frac_at_cap"] = sum(at_cap) / n
            row["frac_at_cap_after_first"] = (
                sum(at_cap[row["first_cap_tick"] - 1:]) / (n - row["first_cap_tick"] + 1)
                if row["first_cap_tick"] else 0.0
            )
            row["rms_at"] = {str(t): round(rms[t - 1], 4) for t in (1, 16, 64, 256, 1000, 2000, 3000, 4000, 6000, 8000) if t <= n}
            row["mode_energy_end"] = mode_energy(model, alphas[-1])
            tc = {}
            for k in (1, 16, 64, 256):
                idx = list(range(n // 2, n - k, max(1, (n // 2) // 200)))
                tc[str(k)] = round(sum(cos(alphas[i], alphas[i + k]) for i in idx) / len(idx), 4)
            row["temporal_cos_alpha_second_half"] = tc
            # B: shared-prefix forks
            forks = {}
            for P, snap in sorted(snaps.items()):
                fa, fh, fr = [], [], []
                for c in conts:
                    m = copy.deepcopy(snap)
                    out = None
                    for j, p in enumerate(c, start=1):
                        out = tick(m, p, P + j)
                    fa.append(out["field"].reshape(-1))
                    fh.append(a_hat_of(m, out["field"]))
                    fr.append(out["field_rms_before"])
                forks[str(P)] = {
                    "mean_pairwise_cos_alpha": round(mean_pairwise_cos(fa), 4),
                    "mean_pairwise_cos_a_hat": round(mean_pairwise_cos(fh), 4),
                    "cos_alpha_vs_prefix_state": round(sum(cos(a, alphas[P - 1]) for a in fa) / len(fa), 4),
                    "rms_before_at_end": [round(x, 3) for x in fr],
                }
            row["shared_prefix_forks"] = forks
            if args.probe:
                start = min(256, n // 4)
                X = torch.stack(ahat[start:n])
                y_next = torch.tensor([pkts[t + 1].payload[0] for t in range(start, n)])
                y_cur = torch.tensor([pkts[t].payload[0] for t in range(start, n)])
                row["probe_next_byte_from_a_hat"] = softmax_probe(X, y_next)
                row["probe_cur_byte_from_a_hat"] = softmax_probe(X, y_cur)
            row["elapsed_s"] = round(time.time() - t0, 1)
            results["rows"].append(row)
            print(json.dumps(row), flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
