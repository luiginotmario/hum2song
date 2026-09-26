#!/usr/bin/env python
"""Hum vs whistle (vs sing) share one melody contour -- paper figure.

Two stages:
  extract : per clip -> npz. Hums/sings: RMVPE (RVC rmvpe.pt, CPU, 10 ms hop), keys f0, conf.
            Whistles: STFT spectral peak (peak_f0, tonal_db, prom; the paper's whistle F0, D-007)
            plus RMVPE at half speed x2 (f0_fix, conf_fix; cross-check only).
  plot    : key-normalize (semitones minus own median), DTW-align every clip to a reference hum
            (contour_pipeline.score_pair: length-normalized, slope-constrained), compute stats,
            draw the figure. Clip lists come from <cache>/clips.json written by extract.

Modes
  MLEnd   : --mlend-root DIR --song Potter   (hum + whistle)
  Custom  : --custom-dir DIR                 (files/subfolders whose name contains
            'hum' / 'whistle' / 'sing', e.g. Happy Birthday recorded by the author)
            -> 3-line version (hum, whistle, sing) automatically.

MLEnd whistle tracks can come from validate_whistle_f0.py's cache (same npz keys) via
--whistle-cache; by default whistle tracks are read from --cache.

Example
  python contour_figure.py extract --mlend-root $R --song Potter --out f0cache --procs 3
  python contour_figure.py plot --cache f0cache_potter --whistle-cache wcache --outdir figs \
      --null-songs StarWars,Hakuna --null-cache-pattern f0cache_{song} --interpreter 213
  python contour_figure.py extract --custom-dir my_hb --out f0cache_hb
  python contour_figure.py plot --cache f0cache_hb --outdir figs_hb --tag happybirthday
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

import contour_pipeline as cp

TYPES = ["hum", "whistle", "sing"]
AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
# Okabe-Ito colorblind-safe palette, one color per query type
COLORS = {"hum": "#0072B2", "whistle": "#D55E00", "sing": "#009E73"}
LABELS = {"hum": "Hum", "whistle": "Whistle", "sing": "Sing"}
WHISTLE_F0_NAMES = {"peak": "spectral peak, tonal ratio >= 6 dB (D-007)",
                    "rmvpe_fix": "RMVPE at half speed, F0 x 2"}
PLOT_STYLE = {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
              "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
              "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
              "ps.fonttype": 42}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ----------------------------------------------------------------------------- clip listing
def custom_clips(custom_dir):
    clips = []
    for p in sorted(glob.glob(os.path.join(custom_dir, "**", "*"), recursive=True)):
        rel = os.path.relpath(p, custom_dir).lower()
        qtype = next((t for t in TYPES if t in rel), None)
        if p.lower().endswith(AUDIO_EXT) and qtype:
            clips.append(dict(path=p, id=os.path.splitext(rel.replace(os.sep, "__"))[0],
                              type=qtype, interpreter="author"))
    return clips


def mlend_clips(mlend_root, song):
    with open(os.path.join(mlend_root, "MLEndHWD_audio_attributes_benchmark.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r["Song"] == song]
    return [dict(path=os.path.join(mlend_root, "MLEndHWD_audiofiles", r["filename"]),
                 id=os.path.splitext(r["filename"])[0], type=r["Interpretation"].lower(),
                 interpreter=str(r["Interpreter"])) for r in rows]


def list_audio_clips(args):
    """Clips (path, id, type, interpreter) whose audio file exists."""
    clips = custom_clips(args.custom_dir) if args.custom_dir else mlend_clips(args.mlend_root,
                                                                             args.song)
    missing = [os.path.basename(c["path"]) for c in clips if not os.path.exists(c["path"])]
    if missing:
        print(f"warning: {len(missing)} listed files missing, skipped:", missing)
    return [c for c in clips if os.path.exists(c["path"])]


def cached_clips(cache):
    with open(os.path.join(cache, "clips.json")) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- extraction
def rmvpe_f0(model, y, speed_down):
    """RMVPE on samples read as 16 kHz; speed_down = 2 means the clip was loaded at 32 kHz (half
    speed, one octave lower), so F0 is doubled and every 2nd frame kept (10 ms hop)."""
    mel = model.extract_mel(y, center=True)
    hidden = model.mel2hidden(mel).squeeze(0).cpu().numpy()
    f0 = model.decode(hidden, thred=0.0) * speed_down
    return f0[::speed_down].astype(np.float32), hidden.max(axis=1)[::speed_down].astype(np.float32)


def voice_arrays(model, path):
    import librosa
    y, _ = librosa.load(path, sr=16000, mono=True)
    f0, conf = rmvpe_f0(model, y, 1)
    return dict(f0=f0, conf=conf)


def whistle_arrays(model, path):
    import librosa
    y, _ = librosa.load(path, sr=cp.SR, mono=True)
    f0_fix, conf_fix = rmvpe_f0(model, y, 2)
    track = cp.spectral_track(y)
    n = min(len(f0_fix), len(track["peak_f0"]))
    keep = {k: track[k][:n].astype(np.float32) for k in ("peak_f0", "tonal_db", "prom")}
    return dict(f0_fix=f0_fix[:n], conf_fix=conf_fix[:n], **keep)


EXTRACTORS = {"hum": voice_arrays, "sing": voice_arrays, "whistle": whistle_arrays}


def extract_worker(job):
    clips, out, model_path, rmvpe_dir, threads = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, rmvpe_dir)
    from rmvpe_rvc import RMVPE
    model = RMVPE(model_path, False, device="cpu")
    for c in clips:
        target = os.path.join(out, c["id"] + ".npz")
        if os.path.exists(target):
            continue
        np.savez_compressed(target, hop=cp.HOP_S, **EXTRACTORS[c["type"]](model, c["path"]))
        print("done", c["id"], flush=True)


def cmd_extract(args):
    os.makedirs(args.out, exist_ok=True)
    clips = list_audio_clips(args)
    jobs = [(clips[i::args.procs], args.out, args.model, args.rmvpe_dir, args.threads)
            for i in range(args.procs)]
    if args.procs == 1:
        extract_worker(jobs[0])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.procs) as pool:
            pool.map(extract_worker, jobs)
    with open(os.path.join(args.out, "clips.json"), "w") as fh:
        json.dump(clips, fh, indent=1)


# ----------------------------------------------------------------------------- contour loading
def voice_track(z):
    f0, conf = z["f0"].astype(float), z["conf"].astype(float)
    voiced = cp.rmvpe_voiced(f0, conf)
    return f0, voiced, float(conf[voiced].mean()) if voiced.any() else 0.0


def peak_whistle_track(z):
    return z["peak_f0"].astype(float), cp.peak_voiced(z), np.nan


def rmvpe_fix_whistle_track(z):
    f0, conf = z["f0_fix"].astype(float), z["conf_fix"].astype(float)
    voiced = cp.rmvpe_voiced(f0, conf)
    return f0, voiced, float(conf[voiced].mean()) if voiced.any() else 0.0


WHISTLE_TRACKS = {"peak": peak_whistle_track, "rmvpe_fix": rmvpe_fix_whistle_track}


def load_track(clip, cache, whistle_cache, whistle_f0):
    """(f0 Hz, voiced mask, mean RMVPE salience on voiced frames or NaN) for one clip, or None
    when its npz is missing."""
    is_whistle = clip["type"] == "whistle"
    path = os.path.join(whistle_cache if is_whistle else cache, clip["id"] + ".npz")
    if not os.path.exists(path):
        return None
    reader = WHISTLE_TRACKS[whistle_f0] if is_whistle else voice_track
    return reader(np.load(path))


def process_clip(clip, track):
    """Per-clip contour and quality features, or None when too few frames are voiced."""
    f0, voiced, mean_conf = track
    normalized = cp.key_normalized(f0, voiced)
    if normalized is None:
        return None
    st, median, norm = normalized
    _, seq = cp.voiced_seq(st)
    return dict(clip, t=np.arange(len(f0)) * cp.HOP_S, st=st, med=median, norm=norm,
                vfrac=float(np.mean(~np.isnan(st))), mconf=mean_conf,
                jump=float(np.median(np.abs(np.diff(seq)))))


def load_processed(clips, cache, whistle_cache, whistle_f0):
    processed = {}
    for c in clips:
        track = load_track(c, cache, whistle_cache, whistle_f0)
        clip = process_clip(c, track) if track is not None else None
        if clip is not None:
            processed[c["id"]] = clip
    return processed


def group_by_interpreter(processed):
    groups = {}
    for p in processed.values():
        groups.setdefault(p["interpreter"], []).append(p)
    return groups


def of_type(clips, qtype):
    return [p for p in clips if p["type"] == qtype]


# ----------------------------------------------------------------------------- pair scoring
class PairScorer:
    """cp.score_pair with memoization by (reference id, query id)."""

    def __init__(self):
        self.cache = {}

    def __call__(self, ref, qry):
        key = (ref["id"], qry["id"])
        if key not in self.cache:
            self.cache[key] = cp.score_pair(ref["norm"], qry["norm"])
        return self.cache[key]


def pair_row(score, **fields):
    """One scored pair; mae/r are None when the pair could not be aligned."""
    aligned = score is not None
    return dict(fields, mae=score.mae if aligned else None, r=score.r if aligned else None,
                length_ratio=score.length_ratio if aligned else None, aligned=aligned)


def same_song_rows(by_int, scorer):
    """Every hum vs every non-hum clip of the same interpreter."""
    rows = []
    for interp, clips in by_int.items():
        for h in of_type(clips, "hum"):
            for q in [c for c in clips if c["type"] != "hum"]:
                rows.append(pair_row(scorer(h, q), interpreter=interp, ref=h["id"], qry=q["id"],
                                     qtype=q["type"], offset=q["med"] - h["med"],
                                     quality=min(h["vfrac"], q["vfrac"])))
    return rows


def ceiling_rows(by_int, scorer):
    """The hums of one interpreter against each other (repeat consistency)."""
    rows = []
    for interp, clips in by_int.items():
        hums = of_type(clips, "hum")
        for a in range(len(hums)):
            for b in range(a + 1, len(hums)):
                rows.append(pair_row(scorer(hums[a], hums[b]), interpreter=interp,
                                     ref=hums[a]["id"], qry=hums[b]["id"], qtype="hum"))
    return rows


def null_rows(by_int, args, scorer):
    """Each interpreter's hums vs their own clips of other songs (performer-matched null)."""
    rows = []
    for song in [s for s in args.null_songs.split(",") if s]:
        cache = args.null_cache_pattern.format(song=song)
        if not os.path.exists(os.path.join(cache, "clips.json")):
            print("null cache missing:", cache)
            continue
        clips = [c for c in cached_clips(cache) if c["interpreter"] in by_int]
        others = load_processed(clips, cache, args.whistle_cache or cache, args.whistle_f0)
        for q in others.values():
            for h in of_type(by_int[q["interpreter"]], "hum"):
                rows.append(pair_row(scorer(h, q), interpreter=q["interpreter"], song=song,
                                     ref=h["id"], qry=q["id"], qtype=q["type"]))
    return rows


# ----------------------------------------------------------------------------- statistics
def aligned_values(rows, key):
    return np.array([x[key] for x in rows if x["aligned"]], float)


def auc_lower_is_better(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((pos < neg).mean() + 0.5 * (pos == neg).mean())


def dtw_counts(rows):
    ratios = [x["length_ratio"] for x in rows if x["aligned"]]
    return dict(n_pairs=len(rows), n_unaligned=sum(not x["aligned"] for x in rows),
                n_length_ratio_outside_half_to_2=sum(cp.outside_tempo_range(r) for r in ratios))


def agreement_summary(rows):
    mae, r = aligned_values(rows, "mae"), aligned_values(rows, "r")
    return dict(dtw_counts(rows), n_interpreters=len({x["interpreter"] for x in rows}),
                mae_median=float(np.median(mae)), mae_mean=float(np.mean(mae)),
                r_median=float(np.median(r)))


def offset_summary(rows):
    offsets = [x["offset"] for x in rows]
    return dict(offset_median=float(np.median(offsets)),
                offset_iqr=[float(np.percentile(offsets, 25)), float(np.percentile(offsets, 75))])


def dataset_summary(rows, ceiling, null):
    summary = {}
    for qtype in sorted({x["qtype"] for x in rows}):
        rr = [x for x in rows if x["qtype"] == qtype]
        summary[qtype] = dict(agreement_summary(rr), **offset_summary(rr))
    if ceiling:
        summary["ceiling_hum_vs_hum_same_performer"] = agreement_summary(ceiling)
    for qtype in sorted({x["qtype"] for x in null}):
        nn = [x for x in null if x["qtype"] == qtype]
        summary[f"null_hum_vs_{qtype}_other_song_same_performer"] = dict(
            agreement_summary(nn), songs=sorted({x["song"] for x in nn}))
    same = aligned_values([x for x in rows if x["qtype"] == "whistle"], "mae")
    other = aligned_values([x for x in null if x["qtype"] == "whistle"], "mae")
    if len(same) and len(other):
        summary["auc_same_vs_other_song_whistle_by_mae"] = auc_lower_is_better(same, other)
    return summary


def performer_ranking(rows, need):
    """Interpreters with a hum pair for every needed type, ranked by mean MAE (best first)."""
    by_int = {}
    for x in [x for x in rows if x["aligned"]]:
        by_int.setdefault(x["interpreter"], []).append(x)
    eligible = {i: xs for i, xs in by_int.items() if {x["qtype"] for x in xs} >= need}
    return sorted(eligible, key=lambda i: np.mean([x["mae"] for x in eligible[i]]))


def auto_featured(rows, by_int, need):
    """Clean + agreeing performer: mean MAE + 0.5 * pitch jitter - min voiced fraction."""
    def score(interp):
        xs = [x for x in rows if x["interpreter"] == interp and x["aligned"]]
        jitter = np.mean([p["jump"] for p in by_int[interp]])
        return np.mean([x["mae"] for x in xs]) + 0.5 * jitter - np.mean([x["quality"] for x in xs])
    return min(performer_ranking(rows, need), key=score)


def featured_selection(by_int, types_present, scorer, feat):
    """Reference hum (most voiced hum), clips per type, and the best-matching clip per type."""
    clips = by_int[feat]
    ref = max(of_type(clips, "hum"), key=lambda p: p["vfrac"] * p["mconf"])
    per_type = {t: of_type(clips, t) for t in types_present}
    primary = {t: min(per_type[t], key=lambda p: scorer(ref, p).mae)
               for t in types_present if t != "hum"}
    stats = {t: dict(clip=p["id"], mae=scorer(ref, p).mae, r=scorer(ref, p).r,
                     offset=p["med"] - ref["med"]) for t, p in primary.items()}
    return ref, per_type, dict(primary, hum=ref), stats


def aligned_contour(ref, p, scorer):
    return ref["norm"] if p is ref else scorer(ref, p).aligned


# ----------------------------------------------------------------------------- plotting
def note_label(midi):
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1} ({440 * 2 ** ((midi - 69) / 12):.0f} Hz)"


def panel_raw_pitch(ax, types_present, primary, feat_stats):
    voiced = []
    for t in types_present:
        ax.plot(primary[t]["t"], primary[t]["st"], color=COLORS[t], lw=0.9)
        voiced.append(primary[t]["st"][~np.isnan(primary[t]["st"])])
    lo = np.floor(min(v.min() for v in voiced)) - 2
    hi = np.ceil(max(v.max() for v in voiced)) + 2
    ax.set_ylim(lo, hi)
    c_notes = [m for m in range(int(lo), int(hi) + 1) if m % 12 == 0]
    ax.set_yticks(c_notes)
    ax.set_yticklabels([note_label(m) for m in c_notes], fontsize=6.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (log scale)")
    ax.set_title("(a) Raw pitch", loc="left", fontsize=8.5, fontweight="bold")
    offsets = [f"{LABELS[t]} {s['offset']:+.1f} st" for t, s in feat_stats.items()]
    ax.text(0.98, 0.5, "median offset vs hum:\n" + "\n".join(offsets), transform=ax.transAxes,
            fontsize=6.5, ha="right", va="center")


def contour_ylim(ref):
    q_lo, q_hi = np.percentile(ref["norm"], [0.5, 99.5])
    return min(q_lo - 3, -8), max(q_hi + 3, 8)


def panel_featured(ax, tt, ref, per_type, primary, feat_stats, scorer, types_present):
    band = np.vstack([aligned_contour(ref, p, scorer) for t in types_present for p in per_type[t]])
    ax.fill_between(tt, band.min(0), band.max(0), color="0.55", alpha=0.3, lw=0, zorder=1)
    for t in types_present:
        ax.plot(tt, aligned_contour(ref, primary[t], scorer), color=COLORS[t], lw=0.9, zorder=3)
    ax.set_ylim(*contour_ylim(ref))
    ax.set_xlabel("Time (s, reference hum)")
    ax.set_ylabel("Semitones re. clip median")
    ax.set_title("(b) Key-normalized + DTW", loc="left", fontsize=8.5, fontweight="bold")
    text = "\n".join(f"{LABELS[t]} vs hum: MAE {s['mae']:.2f} st, r = {s['r']:.2f}"
                     for t, s in feat_stats.items())
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=6.5, va="bottom")
    return len(band)


def panel_performers(ax, tt, ref, interpreters, by_int, scorer, types_present):
    contours = {t: [] for t in types_present}
    for interp in interpreters:
        for p in by_int[interp]:
            a = aligned_contour(ref, p, scorer)
            contours[p["type"]].append(a)
            ax.plot(tt, a, color=COLORS[p["type"]], lw=0.35, alpha=0.3, zorder=2)
    for t in [t for t in types_present if contours[t]]:
        ax.plot(tt, np.median(np.vstack(contours[t]), 0), color=COLORS[t], lw=1.2, zorder=4)
    counts = ", ".join(f"{len(contours[t])} {LABELS[t].lower()}s" for t in types_present)
    ax.text(0.02, 0.02, f"thin: {counts}\nthick: per-type median", transform=ax.transAxes,
            fontsize=6.5, va="bottom")
    ax.set_ylim(*contour_ylim(ref))
    ax.set_xlabel("Time (s, reference hum)")
    ax.set_title(f"(c) {len(interpreters)} performers", loc="left", fontsize=8.5,
                 fontweight="bold")


def panel_c_interpreters(by_int, feat, types_present, n_performers):
    """Featured performer plus the lowest-jitter performers that have every query type."""
    others = [i for i in by_int
              if i != feat and {p["type"] for p in by_int[i]} >= set(types_present)]
    others.sort(key=lambda i: np.mean([p["jump"] for p in by_int[i]]))
    return [feat] + others[:n_performers]


def draw_figure(args, sel, by_int, scorer, types_present, feat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ref, per_type, primary, feat_stats = sel
    n_panels = 3 if (args.panel_c and len(by_int) > 1) else 2
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.16 if n_panels == 3 else 5.2, 2.55),
                             gridspec_kw=dict(width_ratios=[1.1] + [1] * (n_panels - 1)))
    tt = np.arange(len(ref["norm"])) * cp.HOP_S * cp.DS
    panel_raw_pitch(axes[0], types_present, primary, feat_stats)
    n_band = panel_featured(axes[1], tt, ref, per_type, primary, feat_stats, scorer,
                            types_present)
    interpreters = [feat]
    if n_panels == 3:
        interpreters = panel_c_interpreters(by_int, feat, types_present, args.n_performers)
        panel_performers(axes[2], tt, ref, interpreters, by_int, scorer, types_present)
    handles = [Line2D([], [], color=COLORS[t], lw=1.5, label=LABELS[t]) for t in types_present]
    handles.append(Patch(facecolor="0.55", alpha=0.3,
                         label=f"(b) min–max over {n_band} clips of one performer"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.92))
    base = os.path.join(args.outdir, f"fig_contours_{args.tag}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    return base, interpreters


def featured_rank(rows, feat, need):
    ranking = performer_ranking(rows, need)
    return dict(rank_by_mean_mae=ranking.index(feat) + 1, n_ranked=len(ranking),
                top5=ranking[:5])


def cmd_plot(args):
    os.makedirs(args.outdir, exist_ok=True)
    processed = load_processed(cached_clips(args.cache), args.cache,
                               args.whistle_cache or args.cache, args.whistle_f0)
    types_present = [t for t in TYPES if any(p["type"] == t for p in processed.values())]
    need = set(types_present) - {"hum"}
    print("clips processed:", len(processed), "types:", types_present)
    by_int = group_by_interpreter(processed)
    scorer = PairScorer()
    rows = same_song_rows(by_int, scorer)
    summary = dataset_summary(rows, ceiling_rows(by_int, scorer), null_rows(by_int, args, scorer))
    feat = args.interpreter or auto_featured(rows, by_int, need)
    sel = featured_selection(by_int, types_present, scorer, feat)
    base, interpreters = draw_figure(args, sel, by_int, scorer, types_present, feat)
    ref, per_type, _, feat_stats = sel
    feat_pairs = [x for x in rows if x["interpreter"] == feat]
    res = dict(song=args.tag if args.custom_dir else args.song,
               pitch_model=dict(hum="RMVPE (RVC rmvpe.pt), CPU, normal speed",
                                whistle=WHISTLE_F0_NAMES[args.whistle_f0]),
               whistle_f0=args.whistle_f0, conf_threshold=cp.CONF_THR,
               whistle_peak_rule=list(cp.WHISTLE_PEAK_RULE),
               dtw=dict(steps=cp.DTW_STEPS.tolist(), step_weights=cp.DTW_STEP_WEIGHTS.tolist(),
                        band_radius=cp.DTW_BAND_RADIUS, frame_s=cp.HOP_S * cp.DS,
                        length_normalization="query linearly resampled to reference length",
                        fallback="none: unalignable pairs are counted (n_unaligned), not scored"),
               featured_interpreter=feat, featured_selection="user" if args.interpreter else "auto",
               featured_is_best_case=True, reference_hum=ref["id"],
               featured_clips={t: [p["id"] for p in per_type[t]] for t in types_present},
               featured_stats=feat_stats,
               featured_all_pairs=[dict(ref=x["ref"], qry=x["qry"], mae=x["mae"], r=x["r"])
                                   for x in feat_pairs],
               featured_rank=featured_rank(rows, feat, need), dataset_summary=summary,
               panel_c_interpreters=interpreters,
               notes="featured performer is a best case (chosen for recording quality and "
                     "hum/whistle agreement); panel (c) performers chosen by pitch-track jitter "
                     "only; dataset_summary uses all performers")
    with open(base + "_stats.json", "w") as fh:
        json.dump(res, fh, indent=1)
    with open(base + "_pairs.json", "w") as fh:
        json.dump(rows, fh, indent=0)
    print(json.dumps(res, indent=1))


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--mlend-root")
    e.add_argument("--song", default="Potter")
    e.add_argument("--custom-dir", help="folder with hum/whistle/sing clips (names contain type)")
    e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=1)
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--model", default="rmvpe.pt")
    e.add_argument("--rmvpe-dir", default=".")
    p = sub.add_parser("plot")
    p.add_argument("--song", default="Potter")
    p.add_argument("--custom-dir", help="set when the cache came from a --custom-dir extract")
    p.add_argument("--cache", required=True)
    p.add_argument("--whistle-cache", default="",
                   help="npz dir with whistle tracks (peak_f0, tonal_db, f0_fix, conf_fix); "
                        "default: --cache")
    p.add_argument("--whistle-f0", default="peak", choices=sorted(WHISTLE_TRACKS))
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", default="potter")
    p.add_argument("--interpreter", default=None)
    p.add_argument("--null-songs", default="", help="e.g. StarWars,Hakuna (MLEnd mode)")
    p.add_argument("--null-cache-pattern", default="f0cache_{song}")
    p.add_argument("--panel-c", type=int, default=1)
    p.add_argument("--n-performers", type=int, default=7)
    return ap


def main():
    args = build_parser().parse_args()
    {"extract": cmd_extract, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    main()
