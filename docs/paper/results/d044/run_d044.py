"""D-044: closed-set real-whistle experiment on MLEnd (8 songs), performer-disjoint.

Fine-tunes E2b on MLEnd train-performer clips only (whistle -> hum and hum -> hum pairs of
the same song by another train performer; InfoNCE with same-song masking), selects the
checkpoint on val performers (whistle -> hums top-1), and scores test performers with the
D-013 query-by-example protocol (centroid of other people's clips, query's performer left
out, 8-way). 95% CIs: bootstrap over test clips (and, as a stricter check, over test
performers). Low-data curve: nested 10/25/50/100% subsets of the train performers.
Inputs: f0/{rmvpe,peak}_mlend.npz (luigir5/hum2song-data) and the official MLEnd CSV.
CPU only. No synthetic queries are used.
  python run_d044.py --data DIR --ckpt e2b_best.pt --out DIR [--steps N] [--fracs ...] [--seeds ...]
"""

import argparse, csv, json, time
from pathlib import Path

import sys, types

import numpy as np
import torch

try:  # unused on this path; the CPU box venv has no torchaudio
    import torchaudio  # noqa: F401
except ImportError:
    sys.modules['torchaudio'] = types.ModuleType('torchaudio')
from torch.utils.data import ConcatDataset, DataLoader

from hum2song.contour.data import collate_pairs, set_input_kind
from hum2song.contour.example_eval import song_scores
from hum2song.contour.mlend import MLEndClip, MLEndPairDataset, example_set, load_contours, read_split
from hum2song.contour.train import augment_spec, load_checkpoint
from hum2song.losses import info_nce_symmetric

BATCH = 32
CSV = "raw/mlend_hums_whistles/MLEndHWD_audio_attributes.csv"


def clips(data: Path) -> list[MLEndClip]:
    keys = set(np.load(data / "f0/rmvpe_mlend.npz").files)
    out = []
    for row in csv.DictReader(open(data / CSV, encoding="utf-8")):
        path = f"queries/mlend/mlend_{row['filename']}"
        if path in keys:
            out.append(MLEndClip(path, f"mlend:{row['Song']}", row["Interpreter"], row["Interpretation"].lower()))
    return sorted(out)


MAX_TRAIN_FRAMES = 600  # 12 s at 20 ms: E2b's query_max_s; keeps CPU attention memory bounded


class Capped(torch.utils.data.Dataset):
    """Random <=12 s window of the query and of the hum partner (training only)."""

    def __init__(self, base, seed):
        self.base, self.seed, self.epoch = base, seed, base.epoch

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        rng = np.random.default_rng([self.seed, self.epoch.get(), i, 7])
        for key in ("query", "song"):
            x = item[key]
            if len(x) > MAX_TRAIN_FRAMES:
                start = int(rng.integers(0, len(x) - MAX_TRAIN_FRAMES + 1))
                item[key] = x[start : start + MAX_TRAIN_FRAMES]
        return item


def ranks(model, q, r, cont, device):
    qs, rs = example_set(model, q, cont, device), example_set(model, r, cont, device)
    scores, songs = song_scores(qs, rs, "centroid")
    t = np.array([songs.index(s) for s in qs.songs])
    return (scores > scores[np.arange(len(t)), t][:, None]).sum(1), np.array(qs.people)


def boot(rk, people, k, n=10000, seed=0):
    hit = (rk < k).astype(float)
    rng = np.random.default_rng(seed)
    clip = hit[rng.integers(0, len(hit), (n, len(hit)))].mean(1)
    up = sorted(set(people)); idx = {p: np.where(people == p)[0] for p in up}
    perf = []
    for _ in range(2000):
        pick = rng.choice(up, len(up))
        perf.append(np.concatenate([hit[idx[p]] for p in pick]).mean())
    return {"value": float(hit.mean()), "ci95_clip": [float(np.percentile(clip, 2.5)), float(np.percentile(clip, 97.5))],
            "ci95_performer": [float(np.percentile(perf, 2.5)), float(np.percentile(perf, 97.5))], "n": int(len(hit))}


def evaluate(model, groups, cont, device, split):
    w, h = groups[split]["whistle"], groups[split]["hum"]
    out = {}
    for name, q, r in (("whistle->hums", w, h), ("whistle->whistles", w, w), ("hum->hums", h, h)):
        rk, pp = ranks(model, q, r, cont, device)
        out[name] = {"top1": boot(rk, pp, 1), "top3": boot(rk, pp, 3), "ranks": rk.tolist(), "people": pp.tolist()}
    return out


def train(ckpt, groups, cont, train_people, steps, seed, device, log):
    model, cfg, _ = load_checkpoint(ckpt, device)
    set_input_kind(cfg.input_kind)
    torch.manual_seed(seed)
    tw = [c for c in groups["train"]["whistle"] if c.person in train_people]
    th = [c for c in groups["train"]["hum"] if c.person in train_people]
    aug = augment_spec(cfg)
    ds = ConcatDataset([Capped(MLEndPairDataset(tw, th, cont, aug, seed, 1), seed),
                       Capped(MLEndPairDataset(th, th, cont, aug, seed + 1, 1), seed + 1)])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0, collate_fn=collate_pairs,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
    best = (-1.0, 0, None)
    model.eval(); v = val_score(model, groups, cont, device); best = (v, 0, {k: x.clone() for k, x in model.state_dict().items()})
    log(f"  step 0 val whistle->hums top1 {v:.4f}")
    step, epoch = 0, 0
    while step < steps:
        for d in ds.datasets:
            d.epoch.set(epoch)
        for batch in loader:
            model.train(); step += 1
            for g in opt.param_groups: g["lr"] = 3e-5 * min(1.0, step / 50)
            q = model(batch["query"].to(device), batch["query_valid"].to(device))
            s = model(batch["song"].to(device), batch["song_valid"].to(device))
            loss = info_nce_symmetric(q.float(), s.float(), batch["song_id"], model.temperature())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % 100 == 0 or step == steps:
                model.eval(); v = val_score(model, groups, cont, device)
                log(f"  step {step} loss {float(loss):.3f} val whistle->hums top1 {v:.4f}")
                if v > best[0]: best = (v, step, {k: x.clone() for k, x in model.state_dict().items()})
            if step >= steps: break
        epoch += 1
    model.load_state_dict(best[2]); model.eval()
    return model, {"best_val_whistle_top1": best[0], "best_step": best[1], "train_whistles": len(tw), "train_hums": len(th),
                   "train_performers": len(train_people)}


def val_score(model, groups, cont, device):
    with torch.no_grad():
        rk, _ = ranks(model, groups["val"]["whistle"], groups["val"]["hum"], cont, device)
    return float((rk < 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fracs", default="0.1,0.25,0.5,1.0"); ap.add_argument("--seeds", default="0")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8); device = torch.device("cpu")
    logf = open(a.out / "run.log", "a")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    allc, split = clips(a.data), read_split()
    groups = {s: {t: [c for c in allc if split[c.person] == s and c.qtype == t] for t in ("hum", "whistle")} for s in ("train", "val", "test")}
    cont = {**load_contours(a.data, "rmvpe", [c for c in allc if c.qtype == "hum"]),
            **load_contours(a.data, "peak", [c for c in allc if c.qtype == "whistle"])}
    counts = {s: {t: [len(v), len({c.person for c in v})] for t, v in g.items()} for s, g in groups.items()}
    log(f"counts [clips, performers]: {json.dumps(counts)}")
    model, _, _ = load_checkpoint(a.ckpt, device); model.eval()
    with torch.no_grad(): res = {"e2b_zero_shot": evaluate(model, groups, cont, device, "test")}
    log(f"E2b zero-shot test whistle->hums top1 {res['e2b_zero_shot']['whistle->hums']['top1']['value']:.4f}")
    train_people = sorted({c.person for g in groups["train"].values() for c in g})
    for seed in [int(s) for s in a.seeds.split(",")]:
        order = list(np.random.default_rng(20261014 + seed).permutation(train_people))
        for f in [float(x) for x in a.fracs.split(",")]:
            people = set(order[: max(1, round(f * len(order)))])
            t0 = time.time(); log(f"train frac {f} seed {seed}: {len(people)} performers")
            m, info = train(a.ckpt, groups, cont, people, a.steps, seed, device, log)
            with torch.no_grad(): r = evaluate(m, groups, cont, device, "test")
            r["train"] = {**info, "minutes": (time.time() - t0) / 60}
            res[f"ft_frac{f}_s{seed}"] = r
            log(f"  -> test whistle->hums top1 {r['whistle->hums']['top1']['value']:.4f} top3 {r['whistle->hums']['top3']['value']:.4f}")
            (a.out / "results.json").write_text(json.dumps({"counts": counts, "steps": a.steps, "results": res}, indent=1))


if __name__ == "__main__":
    main()
