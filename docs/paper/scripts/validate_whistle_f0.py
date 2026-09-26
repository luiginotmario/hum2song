#!/usr/bin/env python
"""Validate whistle F0 extraction on every MLEnd whistle (CPU only).

Compares, per 10 ms frame, three whistle F0 estimates:
  rmvpe_fix   : RMVPE fed the clip at half speed (load at 32 kHz, treat as 16 kHz), F0 x 2
                -- the cross-check (the figure's whistle F0 before D-007)
  rmvpe_raw   : RMVPE at normal speed (no fix)
  peak        : STFT spectral-peak frequency (argmax of |X| in 200-6000 Hz, parabolic
                interpolation on log-magnitude); the paper's whistle F0 (D-007) is voiced when
                the tonal ratio (energy within +-50 cents of the peak over the rest of the band)
                is >= 6 dB
There is no ground truth in MLEnd, so agreement between two independent estimators
(neural RMVPE vs plain STFT argmax) is what is measured; a `synth` stage checks both
against pure tones of known frequency.

Stages
  extract    : per whistle -> npz with all per-frame tracks and features (multiprocess, nice it)
  analyze    : per-clip metrics CSV, dataset JSON, histogram figure
  downstream : hum-vs-whistle contour agreement (same song vs other songs, AUC with performer-
               bootstrap CIs) with the whistle F0 from each method; contour cleaning and DTW come
               from contour_pipeline.py, the same code contour_figure.py uses
  synth      : synthetic tone sanity check (known F0, noise/breath added)
  humpeak    : on hums, which harmonic the spectral peak sits on (why peak != F0 for voice)

Example (CPU):
  nice -n 15 python validate_whistle_f0.py extract --mlend-root $R --out wcache --procs 8
  python validate_whistle_f0.py analyze --mlend-root $R --cache wcache --outdir out --peak-thr 6
  python validate_whistle_f0.py downstream --cache wcache --hum-cache f0cache_potter --outdir out
  python validate_whistle_f0.py downstream --cache wcache --hum-cache f0cache_StarWars \
      --song StarWars --null-songs Potter,Hakuna --peak-rules prom:50,tonal_db:6 --outdir out
"""
import argparse
import csv
import json
import os
import sys
from functools import partial

import numpy as np

import contour_pipeline as cp
from contour_pipeline import BAND, CONF_THR, FMAX, FMIN, HOP_S, N_FFT, SR, WIN, spectral_track

PEAK_FEAT, PEAK_THR = cp.WHISTLE_PEAK_RULE  # D-007 rule; analyze can sweep and pick another


def peak_voiced(z, feat=PEAK_FEAT, thr=PEAK_THR):
    return cp.peak_voiced(z, (feat, thr))


def st_diff(a, b):
    """Semitone difference 12*log2(a/b) (nan where either <= 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 12.0 * np.log2(np.where(a > 0, a, np.nan) / np.where(b > 0, b, np.nan))
    return d


def rmvpe_track(m, y16k_like, k):
    """RMVPE on samples that are *interpreted* as 16 kHz. k = speed-down factor (1 or 2)."""
    mel = m.extract_mel(y16k_like.astype(np.float32), center=True)
    hidden = m.mel2hidden(mel).squeeze(0).cpu().numpy()
    f0 = m.decode(hidden, thred=0.0) * k
    conf = hidden.max(axis=1)
    return f0[::k], conf[::k]


def list_whistles(root):
    rows = list(csv.DictReader(open(os.path.join(root, "MLEndHWD_audio_attributes_available.csv"))))
    out = []
    for r in rows:
        if r["Interpretation"].lower() != "whistle":
            continue
        p = os.path.join(root, "MLEndHWD_audiofiles", r["filename"])
        out.append(dict(path=p, id=os.path.splitext(r["filename"])[0], song=r["Song"],
                        interpreter=str(r["Interpreter"]), exists=os.path.exists(p)))
    return out


def _worker(job):
    items, out, model_path, rmvpe_dir, threads = job
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch, librosa
    torch.set_num_threads(threads)
    sys.path.insert(0, rmvpe_dir)
    from rmvpe_rvc import RMVPE
    m = RMVPE(model_path, False, device="cpu")
    for it in items:
        o = os.path.join(out, it["id"] + ".npz")
        if os.path.exists(o):
            continue
        try:
            y32, _ = librosa.load(it["path"], sr=SR, mono=True)
            y16, _ = librosa.load(it["path"], sr=16000, mono=True)
            f_fix, c_fix = rmvpe_track(m, y32, 2)   # 32 kHz samples read as 16 kHz = half speed
            f_raw, c_raw = rmvpe_track(m, y16, 1)
            sp = spectral_track(y32)
            n = min(len(f_fix), len(f_raw), len(sp["peak_f0"]))
            np.savez_compressed(o, f0_fix=f_fix[:n].astype(np.float32), conf_fix=c_fix[:n].astype(np.float32),
                                f0_raw=f_raw[:n].astype(np.float32), conf_raw=c_raw[:n].astype(np.float32),
                                **{kk: v[:n].astype(np.float32) for kk, v in sp.items()},
                                dur=len(y32) / SR)
            print("done", it["id"], flush=True)
        except Exception as ex:  # keep going, report at the end
            print("FAIL", it["id"], repr(ex), flush=True)


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    items = list_whistles(a.mlend_root)
    miss = [i["id"] for i in items if not i["exists"]]
    if miss:
        print("missing files (skipped):", miss)
    items = [i for i in items if i["exists"]]
    if a.limit:
        items = items[: a.limit]
    json.dump(items, open(os.path.join(a.out, "clips.json"), "w"), indent=0)
    jobs = [(items[i::a.procs], a.out, a.model, a.rmvpe_dir, a.threads) for i in range(a.procs)]
    if a.procs == 1:
        _worker(jobs[0])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.procs) as pool:
            pool.map(_worker, jobs)


# ----------------------------------------------------------------------------- analysis
def octave_class(d, tol=1.0):
    """For semitone differences d: n = nearest octave multiple, octave error where n != 0 and
    |d - 12n| < tol."""
    n = np.round(d / 12.0)
    return n, (n != 0) & (np.abs(d - 12 * n) < tol)


def clip_metrics(z, pfeat, pthr):
    f_fix, c_fix, f_raw, c_raw, pk = z["f0_fix"], z["conf_fix"], z["f0_raw"], z["conf_raw"], z["peak_f0"]
    v_fix = (c_fix >= CONF_THR) & (f_fix > FMIN) & (f_fix < FMAX)
    v_raw = (c_raw >= CONF_THR) & (f_raw > FMIN) & (f_raw < FMAX)
    v_pk = peak_voiced(z, pfeat, pthr)
    r = {}
    r["n_frames"] = int(len(f_fix)); r["dur_s"] = float(z["dur"])
    r["voiced_frac_fix"] = float(v_fix.mean()); r["voiced_frac_raw"] = float(v_raw.mean())
    r["voiced_frac_peak"] = float(v_pk.mean())
    r["n_voiced_fix"] = int(v_fix.sum())
    d = st_diff(f_fix, pk)[v_fix]
    dr = st_diff(f_raw, pk)[v_fix]         # no-fix RMVPE on the same frames
    dr_own = st_diff(f_raw, pk)[v_raw]     # no-fix RMVPE on its own voiced frames
    def fr(x, cond):
        return float(np.mean(cond)) if len(x) else np.nan
    r["fix_within05"] = fr(d, np.abs(d) < 0.5); r["fix_within1"] = fr(d, np.abs(d) < 1.0)
    n, oe = octave_class(d); r["fix_oct_err"] = fr(d, oe); r["fix_oct_down"] = fr(d, oe & (n < 0))
    r["fix_oct_up"] = fr(d, oe & (n > 0))
    r["raw_within05"] = fr(dr, np.abs(dr) < 0.5); r["raw_within1"] = fr(dr, np.abs(dr) < 1.0)
    n, oe = octave_class(dr); r["raw_oct_err"] = fr(dr, oe); r["raw_oct_down"] = fr(dr, oe & (n < 0))
    r["raw_own_within05"] = fr(dr_own, np.abs(dr_own) < 0.5)
    n, oe = octave_class(dr_own); r["raw_own_oct_err"] = fr(dr_own, oe)
    r["raw_mean_conf_on_fixvoiced"] = float(c_raw[v_fix].mean()) if v_fix.any() else np.nan
    r["fix_mean_conf"] = float(c_fix[v_fix].mean()) if v_fix.any() else np.nan
    # peak-only as the F0: agreement on frames voiced by the peak detector, and voicing agreement
    both = v_pk & v_fix
    dp = st_diff(pk, f_fix)[both]
    r["peak_within05_on_both"] = fr(dp, np.abs(dp) < 0.5)
    r["peak_voicing_precision"] = float(both.sum() / max(v_pk.sum(), 1))  # vs RMVPE-fix voicing
    r["peak_voicing_recall"] = float(both.sum() / max(v_fix.sum(), 1))
    # features (voiced frames of rmvpe_fix)
    vv = v_fix if v_fix.any() else np.ones_like(v_fix)
    r["median_f0_fix_hz"] = float(np.median(f_fix[vv])); r["median_peak_hz"] = float(np.median(pk[vv]))
    r["median_prom_db"] = float(np.median(z["prom"][vv]))
    r["median_tonal_db"] = float(np.median(z["tonal_db"][vv]))
    r["median_flatness"] = float(np.median(z["flat"][vv]))
    r["median_h2_db"] = float(np.median(z["h2_db"][vv]))
    lev = z["level_db"]
    noise = np.percentile(lev, 10)
    r["snr_db"] = float(np.median(lev[vv]) - noise)  # voiced level over quietest-10% frames
    return r


def classify_failure(r, thr):
    """Heuristic failure labels (thresholds in thr, reported in the JSON)."""
    labs = []
    if r["n_voiced_fix"] < thr["min_voiced"]:
        labs.append("few_voiced_frames")
    if r["fix_oct_err"] > thr["octave"]:
        labs.append("residual_octave_error")
    if r["snr_db"] < thr["snr"]:
        labs.append("low_snr")
    if r["median_tonal_db"] < thr["tonal"]:
        labs.append("breathy_noisy")
    if r["median_f0_fix_hz"] > thr["high_hz"] or r["median_peak_hz"] > thr["high_hz"]:
        labs.append("very_high_pitch")
    return labs or ["unexplained"]


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    os.makedirs(a.outdir, exist_ok=True)
    clips = {c["id"]: c for c in json.load(open(os.path.join(a.cache, "clips.json")))}
    # extract writes only existing files to clips.json; with --mlend-root we also count missing ones
    all_items = list_whistles(a.mlend_root) if a.mlend_root else list(clips.values())
    Z = {}
    for cid in clips:
        f = os.path.join(a.cache, cid + ".npz")
        if os.path.exists(f):
            Z[cid] = dict(np.load(f))
    print("clips with tracks:", len(Z), "of", len(all_items))

    # --- choose the peak voicing rule: sweep two features, report voicing P/R/F1 vs RMVPE-fix voicing
    sweep = []
    grid = [("prom", t) for t in range(20, 75, 5)] + [("tonal_db", t) for t in range(-12, 22, 3)]
    for feat, t in grid:
        tp = fp = fn = 0; agree = []
        for z in Z.values():
            v_fix = (z["conf_fix"] >= CONF_THR) & (z["f0_fix"] > FMIN) & (z["f0_fix"] < FMAX)
            v_pk = peak_voiced(z, feat, t)
            tp += int((v_pk & v_fix).sum()); fp += int((v_pk & ~v_fix).sum()); fn += int((~v_pk & v_fix).sum())
            d = st_diff(z["peak_f0"], z["f0_fix"])[v_pk & v_fix]
            if len(d): agree.append(np.abs(d) < 0.5)
        ag = np.concatenate(agree) if agree else np.array([])
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        sweep.append(dict(feature=feat, threshold=t, precision_vs_rmvpe=P, recall_vs_rmvpe=R,
                          f1=2 * P * R / max(P + R, 1e-9),
                          frames_within05_on_both=float(ag.mean()) if len(ag) else None))
    best = max(sweep, key=lambda s: s["f1"])
    pfeat, pthr = (a.peak_feat, a.peak_thr) if a.peak_thr is not None else (best["feature"], best["threshold"])
    prom_thr = dict(feature=pfeat, threshold=pthr)

    thr = dict(min_voiced=100, octave=0.10, snr=15.0, tonal=0.0, high_hz=2000.0)  # 2 kHz: see synth stage
    rows = []
    for cid, z in Z.items():
        r = dict(id=cid, song=clips[cid]["song"], interpreter=clips[cid]["interpreter"])
        r.update(clip_metrics(z, pfeat, pthr))
        r["failure"] = "" if (r["fix_within05"] >= 0.9) else "+".join(classify_failure(r, thr))
        rows.append(r)
    rows.sort(key=lambda r: (np.nan_to_num(r["fix_within05"], nan=-1)))
    keys = list(rows[0].keys())
    with open(os.path.join(a.outdir, "whistle_f0_validation_clips.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    ok = [r for r in rows if r["n_voiced_fix"] >= thr["min_voiced"]]
    def S(key, rr=ok):
        x = np.array([r[key] for r in rr], float); x = x[~np.isnan(x)]
        return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.percentile(x, 25)),
                    q75=float(np.percentile(x, 75)), mean=float(np.mean(x)),
                    frac_over_0p9=float(np.mean(x > 0.9)), frac_over_0p8=float(np.mean(x > 0.8)))
    # frame-pooled numbers
    def pooled(cond_fn):
        num = den = 0
        for cid in [r["id"] for r in ok]:
            z = Z[cid]
            v = (z["conf_fix"] >= CONF_THR) & (z["f0_fix"] > FMIN) & (z["f0_fix"] < FMAX)
            c = cond_fn(z, v); num += int(c.sum()); den += int(v.sum())
        return num / max(den, 1)
    pooled_fix05 = pooled(lambda z, v: np.abs(st_diff(z["f0_fix"], z["peak_f0"])[v]) < 0.5)
    pooled_raw05 = pooled(lambda z, v: np.abs(st_diff(z["f0_raw"], z["peak_f0"])[v]) < 0.5)
    pooled_fix_oe = pooled(lambda z, v: octave_class(st_diff(z["f0_fix"], z["peak_f0"])[v])[1])
    pooled_raw_oe = pooled(lambda z, v: octave_class(st_diff(z["f0_raw"], z["peak_f0"])[v])[1])

    fails = [r for r in ok if r["fix_within05"] < 0.9]
    from collections import Counter
    lab_counts = Counter(l for r in fails for l in r["failure"].split("+"))
    worst = [dict(id=r["id"], song=r["song"], interpreter=r["interpreter"],
                  fix_within05=r["fix_within05"], fix_within1=r["fix_within1"], fix_oct_err=r["fix_oct_err"],
                  snr_db=r["snr_db"], median_tonal_db=r["median_tonal_db"], median_f0_fix_hz=r["median_f0_fix_hz"],
                  median_peak_hz=r["median_peak_hz"], voiced_frac_fix=r["voiced_frac_fix"], failure=r["failure"])
             for r in ok[:25]]
    # feature contrast pass vs fail
    def med(key, rr):
        return float(np.nanmedian([r[key] for r in rr])) if rr else None
    passed = [r for r in ok if r["fix_within05"] >= 0.9]
    contrast = {k: dict(pass_median=med(k, passed), fail_median=med(k, fails))
                for k in ["snr_db", "median_tonal_db", "median_flatness", "median_prom_db", "median_h2_db",
                          "median_f0_fix_hz", "voiced_frac_fix", "fix_mean_conf"]}
    pk_med = np.array([r["median_peak_hz"] for r in ok]); w05 = np.array([r["fix_within05"] for r in ok])
    by_pitch = {}
    for lo, hi in [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 1e5)]:
        msk = (pk_med >= lo) & (pk_med < hi)
        if msk.any():
            by_pitch[f"{lo:.0f}-{hi:.0f}Hz"] = dict(n=int(msk.sum()), fix_within05_median=float(np.median(w05[msk])),
                                                   frac_over_0p9=float(np.mean(w05[msk] > 0.9)))
    exc = [r for r in rows if r["n_voiced_fix"] < thr["min_voiced"]]
    exc_pk = []
    for r in exc:
        z = Z[r["id"]]; vp = peak_voiced(z, pfeat, pthr)
        exc_pk.append((float(vp.mean()), float(np.median(z["peak_f0"][vp])) if vp.any() else np.nan))
    exc_pk = np.array(exc_pk) if exc_pk else np.zeros((0, 2))
    excluded_summary = dict(n=len(exc), peak_voiced_frac_median=float(np.nanmedian(exc_pk[:, 0])) if len(exc) else None,
                            peak_median_hz_median=float(np.nanmedian(exc_pk[:, 1])) if len(exc) else None,
                            frac_peak_median_over_2khz=float(np.nanmean(exc_pk[:, 1] > 2000)) if len(exc) else None,
                            n_peak_voiced_under_5pct=int(np.sum(exc_pk[:, 0] < 0.05)) if len(exc) else None,
                            note="RMVPE-fix voices < 100 frames; mostly very high whistles (above RMVPE-fix's ~4 kHz ceiling) "
                                 "or clips with little whistling")
    fail_pk = np.array([r["median_peak_hz"] for r in fails])
    # where do disagreements sit? pooled over voiced frames of analyzed clips (d = RMVPE-fix - peak, st)
    dd, sub_fail_oct, sub_pass = [], [], []
    fail_ids = {r["id"] for r in fails}
    for r in ok:
        z = Z[r["id"]]
        v = (z["conf_fix"] >= CONF_THR) & (z["f0_fix"] > FMIN) & (z["f0_fix"] < FMAX)
        d = st_diff(z["f0_fix"], z["peak_f0"])[v]; dd.append(d)
        if r["id"] in fail_ids:
            oct_dn = np.abs(d + 12) < 1
            if oct_dn.sum() >= 20:
                sub_fail_oct.append(float(np.median(z["sub_db"][v][oct_dn])))
        elif v.any():
            sub_pass.append(float(np.median(z["sub_db"][v])))
    dd = np.concatenate(dd); tot = len(dd)
    disagreement = dict(n_frames=tot, within05=float(np.mean(np.abs(dd) < 0.5)),
                        near_minus12=float(np.mean(np.abs(dd + 12) < 1)), near_minus24=float(np.mean(np.abs(dd + 24) < 1)),
                        near_plus12=float(np.mean(np.abs(dd - 12) < 1)),
                        near_minus19p02_peak_is_3x=float(np.mean(np.abs(dd + 19.02) < 1)),
                        near_plus19p02=float(np.mean(np.abs(dd - 19.02) < 1)))
    disagreement["other"] = 1 - sum(v for k, v in disagreement.items() if k not in ("n_frames",)) + 0.0
    disagreement["level_at_half_peak_freq_db"] = dict(
        failing_clips_octave_down_frames_median=float(np.median(sub_fail_oct)) if sub_fail_oct else None,
        n_failing_clips=len(sub_fail_oct), passing_clips_median=float(np.median(sub_pass)) if sub_pass else None,
        note="spectral level at f_peak/2 relative to the peak; if RMVPE's octave-down value were the true F0, "
             "a component would be expected there")
    by_song = {}
    for s in sorted({r["song"] for r in ok}):
        rr = [r for r in ok if r["song"] == s]
        by_song[s] = dict(n=len(rr), fix_within05_median=med("fix_within05", rr),
                          frac_over_0p9=float(np.mean([r["fix_within05"] > 0.9 for r in rr])))
    # spearman correlations of agreement with features
    from scipy.stats import spearmanr
    corr = {k: float(spearmanr([r[k] for r in ok], [r["fix_within05"] for r in ok], nan_policy="omit")[0])
            for k in ["snr_db", "median_tonal_db", "median_flatness", "median_prom_db", "median_f0_fix_hz",
                      "voiced_frac_fix", "fix_mean_conf"]}
    out = dict(
        n_whistles_listed=len(all_items), n_files_missing=sum(not i["exists"] for i in all_items),
        missing_ids=[i["id"] for i in all_items if not i["exists"]],
        n_extracted=len(Z), n_analyzed=len(ok),
        excluded_few_voiced=[r["id"] for r in rows if r["n_voiced_fix"] < thr["min_voiced"]],
        settings=dict(conf_thr=CONF_THR, f0_limits_hz=[FMIN, FMAX], stft=dict(sr=SR, n_fft=N_FFT, win=WIN,
                      hop_s=HOP_S, band_hz=BAND, interp="parabolic on dB"), peak_voicing_rule=prom_thr,
                      octave_error_def="|d-12n|<1 st for integer n!=0, d=12log2(f_est/f_peak)",
                      failure_thresholds=thr, failure_def="clip with fix_within05 < 0.9"),
        per_clip_fix_within05=S("fix_within05"), per_clip_fix_within1=S("fix_within1"),
        per_clip_fix_octave_err=S("fix_oct_err"),
        per_clip_raw_within05_same_frames=S("raw_within05"), per_clip_raw_within1_same_frames=S("raw_within1"),
        per_clip_raw_octave_err_same_frames=S("raw_oct_err"), per_clip_raw_octave_down_same_frames=S("raw_oct_down"),
        per_clip_raw_within05_own_voicing=S("raw_own_within05"), per_clip_raw_octave_err_own_voicing=S("raw_own_oct_err"),
        per_clip_voiced_frac=dict(fix=S("voiced_frac_fix"), raw=S("voiced_frac_raw"), peak=S("voiced_frac_peak")),
        pooled_frames=dict(fix_within05=pooled_fix05, raw_within05=pooled_raw05,
                           fix_octave_err=pooled_fix_oe, raw_octave_err=pooled_raw_oe),
        peak_only=dict(threshold_sweep=sweep, chosen_rule=prom_thr, chosen_by="max F1 vs RMVPE-fix voicing" if a.peak_thr is None else "user",
                       per_clip_within05_on_both_voiced=S("peak_within05_on_both"),
                       per_clip_voicing_precision_vs_rmvpe=S("peak_voicing_precision"),
                       per_clip_voicing_recall_vs_rmvpe=S("peak_voicing_recall")),
        median_whistle_f0_hz=dict(rmvpe_fix=float(np.median([r["median_f0_fix_hz"] for r in ok])),
                                  peak=float(np.median([r["median_peak_hz"] for r in ok]))),
        by_pitch_band_of_spectral_peak=by_pitch, excluded_clips=excluded_summary,
        frame_disagreement_fix_vs_peak=disagreement,
        failures=dict(n=len(fails), frac=len(fails) / max(len(ok), 1), label_counts=dict(lab_counts),
                      frac_with_median_peak_over_2khz=float(np.mean(fail_pk > 2000)) if len(fails) else None,
                      top_interpreters=sorted(Counter(r["interpreter"] for r in fails).items(), key=lambda x: -x[1])[:10],
                      feature_contrast=contrast, spearman_with_fix_within05=corr, worst_25=worst),
        by_song=by_song,
    )
    json.dump(out, open(os.path.join(a.outdir, "whistle_f0_validation.json"), "w"), indent=1)

    # --- histogram figure
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    bins = np.linspace(0, 100, 41)
    x_fix = 100 * np.array([r["fix_within05"] for r in ok]); x_raw = 100 * np.array([r["raw_within05"] for r in ok])
    ax[0].hist(x_raw, bins=bins, color="#999999", alpha=0.8, label="RMVPE, normal speed")
    ax[0].hist(x_fix, bins=bins, color="#D55E00", alpha=0.8, label="RMVPE, half speed, F0\u00d72")
    ax[0].axvline(90, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("voiced frames within 0.5 st of spectral peak (%)"); ax[0].set_ylabel("whistle clips")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center")
    ax[0].set_title(f"(a) per clip, n = {len(ok)}", fontsize=9, loc="left")
    dall = []
    for r in ok:
        z = Z[r["id"]]
        v = (z["conf_fix"] >= CONF_THR) & (z["f0_fix"] > FMIN) & (z["f0_fix"] < FMAX)
        dall.append(np.stack([st_diff(z["f0_fix"], z["peak_f0"])[v], st_diff(z["f0_raw"], z["peak_f0"])[v]]))
    dall = np.concatenate(dall, 1)
    b2 = np.linspace(-30, 30, 241)
    ax[1].hist(np.clip(dall[1], -30, 30), bins=b2, color="#999999", alpha=0.8, label="normal speed")
    ax[1].hist(np.clip(dall[0], -30, 30), bins=b2, color="#D55E00", alpha=0.8, label="half speed, \u00d72")
    ax[1].set_yscale("log"); ax[1].set_xlabel("RMVPE \u2212 spectral peak (semitones, per frame)")
    ax[1].set_ylabel("frames"); ax[1].legend(frameon=False, fontsize=8)
    for xv in (-24, -12, 12):
        ax[1].axvline(xv, color="k", ls=":", lw=0.6)
    ax[1].set_title(f"(b) all voiced frames, n = {dall.shape[1]:,}", fontsize=9, loc="left")
    ax[2].scatter(pk_med, 100 * w05, s=4, color="#D55E00", alpha=0.5, lw=0)
    ax[2].set_xscale("log"); ax[2].set_xticks([500, 1000, 2000, 4000]); ax[2].set_xticklabels(["500", "1k", "2k", "4k"])
    ax[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[2].axvline(2000, color="k", ls=":", lw=0.8)
    ax[2].set_xlabel("clip median spectral-peak frequency (Hz)"); ax[2].set_ylabel("frames within 0.5 st (%)")
    ax[2].set_title("(c) agreement vs whistle pitch", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "fig_whistle_f0_validation.png"), dpi=300)
    fig.savefig(os.path.join(a.outdir, "fig_whistle_f0_validation.pdf"))
    print(json.dumps({k: out[k] for k in ["n_analyzed", "per_clip_fix_within05", "per_clip_raw_within05_same_frames",
                                          "pooled_frames"]}, indent=1))


# ----------------------------------------------------------------------------- downstream check
def auc(pos, neg, lower_is_better=True):
    """P(pos beats neg), ties 0.5; NaN when either side is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    wins = (pos < neg) if lower_is_better else (pos > neg)
    return float(wins.mean() + 0.5 * (pos == neg).mean())


def rmvpe_fix_track(z, rule):
    return z["f0_fix"], cp.rmvpe_voiced(z["f0_fix"], z["conf_fix"])


def rmvpe_raw_track(z, rule):
    return z["f0_raw"], cp.rmvpe_voiced(z["f0_raw"], z["conf_raw"])


def peak_track(z, rule):
    return z["peak_f0"], cp.peak_voiced(z, rule)


def peak_f0_fix_voicing_track(z, rule):
    """Control: spectral-peak F0 values on RMVPE-fix's voiced frames."""
    return z["peak_f0"], cp.rmvpe_voiced(z["f0_fix"], z["conf_fix"]) & (z["peak_f0"] > FMIN)


def fix_f0_peak_voicing_track(z, rule):
    """Control: RMVPE-fix F0 values on the spectral peak's voiced frames."""
    f0 = z["f0_fix"]
    return f0, cp.peak_voiced(z, rule) & (f0 > FMIN) & (f0 < FMAX)


def parse_rule(spec):
    feature, threshold = spec.split(":")
    return feature, float(threshold)


def downstream_methods(peak_rules, control_rule):
    """name -> (track function, spectral-peak voicing rule). The two controls separate "which F0
    values" from "which frames are voiced"."""
    methods = {"rmvpe_fix": (rmvpe_fix_track, None), "rmvpe_raw": (rmvpe_raw_track, None)}
    for feature, threshold in peak_rules:
        methods[f"peak_{feature}{threshold:g}"] = (peak_track, (feature, threshold))
    methods["peakF0_fixVoicing"] = (peak_f0_fix_voicing_track, None)
    feature, threshold = control_rule
    methods[f"fixF0_peakVoicing_{feature}{threshold:g}"] = (fix_f0_peak_voicing_track, control_rule)
    return methods


def load_hum_sequences(hum_cache):
    """interpreter -> [(hum id, key-normalized sequence)] from RMVPE hum tracks."""
    with open(os.path.join(hum_cache, "clips.json")) as fh:
        clips = [c for c in json.load(fh) if c["type"] == "hum"]
    hums = {}
    for c in clips:
        z = np.load(os.path.join(hum_cache, c["id"] + ".npz"))
        f0, conf = z["f0"].astype(float), z["conf"].astype(float)
        normalized = cp.key_normalized(f0, cp.rmvpe_voiced(f0, conf))
        if normalized is not None:
            hums.setdefault(c["interpreter"], []).append((c["id"], normalized[2]))
    return hums


def scored_whistle_pairs(whistles, cache, hums, track, rule, target_song):
    """(hum id, whistle id) -> pair record for every hum of the whistle's performer."""
    pairs = {}
    for c in whistles:
        f0, voiced = track(np.load(os.path.join(cache, c["id"] + ".npz")), rule)
        normalized = cp.key_normalized(np.asarray(f0, float), voiced)
        if normalized is None:
            continue
        for hum_id, hum in hums[c["interpreter"]]:
            score = cp.score_pair(hum, normalized[2])
            pairs[(hum_id, c["id"])] = dict(
                same=c["song"] == target_song, interpreter=c["interpreter"], song=c["song"],
                length_ratio=len(normalized[2]) / len(hum), aligned=score is not None,
                mae=score.mae if score else None, r=score.r if score else None)
    return pairs


def aligned_keys(pairs):
    return [k for k, p in pairs.items() if p["aligned"]]


def auc_of(pairs, keys, value="mae"):
    pos = [pairs[k][value] for k in keys if pairs[k]["same"]]
    neg = [pairs[k][value] for k in keys if not pairs[k]["same"]]
    return auc(pos, neg, lower_is_better=value == "mae")


def auc_difference(pairs_a, pairs_b, keys):
    return auc_of(pairs_a, keys) - auc_of(pairs_b, keys)


def median_of(rows, key):
    return float(np.median([p[key] for p in rows]))


def keys_by_performer(pairs, keys):
    groups = {}
    for k in keys:
        groups.setdefault(pairs[k]["interpreter"], []).append(k)
    return groups


def performer_bootstrap(groups, statistic, n_boot, seed=0):
    """95% percentile interval of statistic(keys) over performer-level resamples."""
    rng = np.random.default_rng(seed)
    performers = sorted(groups)
    values = [statistic([k for p in rng.choice(performers, len(performers)) for k in groups[p]])
              for _ in range(n_boot)]
    return values, [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def method_summary(pairs, n_boot):
    keys = aligned_keys(pairs)
    same = [pairs[k] for k in keys if pairs[k]["same"]]
    other = [pairs[k] for k in keys if not pairs[k]["same"]]
    _, ci = performer_bootstrap(keys_by_performer(pairs, keys), lambda ks: auc_of(pairs, ks),
                                n_boot)
    return dict(n_same=len(same), n_other=len(other),
                mae_median_same=median_of(same, "mae"), r_median_same=median_of(same, "r"),
                mae_median_other=median_of(other, "mae"), r_median_other=median_of(other, "r"),
                auc_by_mae=auc_of(pairs, keys), auc_by_mae_ci95=ci,
                auc_by_r=auc_of(pairs, keys, "r"),
                n_pairs=len(pairs), n_unaligned=len(pairs) - len(keys),
                n_pairs_length_ratio_outside_half_to_2=sum(
                    cp.outside_tempo_range(p["length_ratio"]) for p in pairs.values()))


def paired_auc_difference(pairs_a, pairs_b, keys, n_boot):
    """AUC(a) - AUC(b) on the same pairs, with a performer-bootstrap 95% CI."""
    diff = partial(auc_difference, pairs_a, pairs_b)
    values, ci = performer_bootstrap(keys_by_performer(pairs_b, keys), diff, n_boot)
    performers = {pairs_b[k]["interpreter"] for k in keys}
    return dict(n_common_pairs=len(keys), n_performers=len(performers),
                auc_method=auc_of(pairs_a, keys), auc_rmvpe_fix=auc_of(pairs_b, keys),
                diff=diff(keys), ci95=ci,
                frac_boot_diff_le_0=float(np.mean(np.array(values) <= 0)))


def common_aligned_keys(pair_sets):
    return sorted(set.intersection(*[set(aligned_keys(p)) for p in pair_sets]))


def paired_comparisons(pairs, n_boot):
    """Every method minus rmvpe_fix. Non-raw methods share one common pair set (pairs all of them
    can score); rmvpe_raw uses the pairs it and rmvpe_fix can both score."""
    scored = [m for m in pairs if m != "rmvpe_raw"]
    common = common_aligned_keys([pairs[m] for m in scored])
    out = {f"{m}_minus_rmvpe_fix": paired_auc_difference(pairs[m], pairs["rmvpe_fix"], common,
                                                          n_boot)
           for m in scored if m != "rmvpe_fix"}
    raw_common = common_aligned_keys([pairs["rmvpe_raw"], pairs["rmvpe_fix"]])
    out["rmvpe_raw_minus_rmvpe_fix"] = paired_auc_difference(pairs["rmvpe_raw"],
                                                             pairs["rmvpe_fix"], raw_common, n_boot)
    return out


def write_pairs_csv(path, pairs):
    fields = ["method", "hum", "whistle", "song", "interpreter", "same_song", "aligned",
              "length_ratio", "mae", "r"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for method, pp in pairs.items():
            for (hum, whistle), p in sorted(pp.items()):
                w.writerow([method, hum, whistle, p["song"], p["interpreter"], int(p["same"]),
                            int(p["aligned"]), f"{p['length_ratio']:.4f}",
                            "" if p["mae"] is None else f"{p['mae']:.4f}",
                            "" if p["r"] is None else f"{p['r']:.4f}"])


def cmd_downstream(a):
    """Target-song hums (RMVPE hum cache) vs same-performer whistles of the target song and of the
    null songs, with the whistle F0 from each method. Same pipeline as contour_figure.py
    (contour_pipeline: cleaning, key normalization, length-normalized slope-constrained DTW)."""
    with open(os.path.join(a.cache, "clips.json")) as fh:
        clips = json.load(fh)
    hums = load_hum_sequences(a.hum_cache)
    songs = {a.song} | set(a.null_songs.split(","))
    whistles = [c for c in clips if c["interpreter"] in hums and c["song"] in songs
                and os.path.exists(os.path.join(a.cache, c["id"] + ".npz"))]
    methods = downstream_methods([parse_rule(s) for s in a.peak_rules.split(",")],
                                 parse_rule(a.control_rule))
    res, pairs = {}, {}
    for name, (track, rule) in methods.items():
        pairs[name] = scored_whistle_pairs(whistles, a.cache, hums, track, rule, a.song)
        res[name] = method_summary(pairs[name], a.n_boot)
        print(name, res[name], flush=True)
    res["paired_auc_by_mae_common_pairs"] = paired_comparisons(pairs, a.n_boot)
    res["settings"] = dict(song=a.song, null_songs=sorted(songs - {a.song}),
                           peak_rules=a.peak_rules, control_rule=a.control_rule,
                           hum_cache=a.hum_cache, n_boot=a.n_boot,
                           dtw="query resampled to reference length, then slope-constrained DTW "
                               "(steps (1,1),(1,2),(2,1), band 25%); no fallback, unalignable "
                               "pairs counted in n_unaligned",
                           ci="95% percentile interval, performer-level bootstrap")
    suffix = "" if a.song == "Potter" else f"_{a.song}"
    with open(os.path.join(a.outdir, f"whistle_f0_downstream{suffix}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_pairs_csv(os.path.join(a.outdir, f"whistle_f0_downstream_pairs{suffix}.csv"), pairs)


# ----------------------------------------------------------------------------- synthetic check
def cmd_synth(a):
    """Known-F0 check: whistle-like tones (vibrato glides, weak 2nd harmonic, breath noise)."""
    import torch, librosa
    torch.set_num_threads(a.threads)
    sys.path.insert(0, a.rmvpe_dir)
    from rmvpe_rvc import RMVPE
    m = RMVPE(a.model, False, device="cpu")
    rng = np.random.default_rng(0)
    res = []
    for f_c in [500, 700, 1000, 1400, 2000, 2800, 3500, 4000]:
        for snr in [30, 10, 0]:
            dur = 3.0; t = np.arange(int(dur * SR)) / SR
            f = f_c * 2 ** ((2 * np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 5 * t)) / 12)  # +-2 st glide
            ph = 2 * np.pi * np.cumsum(f) / SR
            y = np.sin(ph) + 0.05 * np.sin(2 * ph)
            noise = rng.standard_normal(len(y)); noise = librosa.effects.preemphasis(noise, coef=-0.9)  # pink-ish
            noise *= np.sqrt(np.mean(y ** 2) / np.mean(noise ** 2)) * 10 ** (-snr / 20)
            y = (0.3 * (y + noise) / np.max(np.abs(y + noise))).astype(np.float32)
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
            ff, cf_ = rmvpe_track(m, y, 2); fr, cr = rmvpe_track(m, y16, 1); sp = spectral_track(y)
            n = min(len(ff), len(fr), len(sp["peak_f0"]))
            ft = f[(np.arange(n) * int(SR * HOP_S)).clip(0, len(f) - 1)]
            sl = slice(5, n - 5)
            def acc(est, v):
                d = st_diff(est[:n][sl], ft[sl]); v = v[:n][sl]
                return float(np.mean(np.abs(d[v]) < 0.5)) if v.any() else None, float(v.mean())
            vf = (cf_ >= CONF_THR) & (ff > FMIN) & (ff < FMAX); vr = (cr >= CONF_THR) & (fr > FMIN) & (fr < FMAX)
            vp = peak_voiced(sp, a.peak_feat, a.peak_thr)
            row = dict(f_center=f_c, snr_db=snr)
            for name, (est, v) in dict(rmvpe_fix=(ff, vf), rmvpe_raw=(fr, vr), peak=(sp["peak_f0"], vp)).items():
                w, vfrac = acc(est, v)
                row[name + "_within05"] = w; row[name + "_voiced"] = vfrac
            res.append(row); print(row, flush=True)
    json.dump(res, open(os.path.join(a.outdir, "whistle_f0_synth.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- hums: is the spectral peak F0?
def cmd_humpeak(a):
    """For hums (existing RMVPE cache from contour_figure.py, normal speed), locate the per-frame
    spectral peak in 50-6000 Hz and report which harmonic number it sits on (peak/F0)."""
    import librosa
    clips = [c for c in json.load(open(os.path.join(a.hum_cache, "clips.json"))) if c["type"] == "hum"]
    if a.limit:
        clips = clips[: a.limit]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / 16000)
    lo, hi = np.searchsorted(freqs, 50.0), np.searchsorted(freqs, 6000.0)
    harm_counts = np.zeros(12); per_clip = []; f0s = []
    for c in clips:
        z = np.load(os.path.join(a.hum_cache, c["id"] + ".npz"))
        f0, conf = z["f0"].astype(float), z["conf"].astype(float)
        y, _ = librosa.load(c["path"], sr=16000, mono=True)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=160, win_length=1024, center=True)).T[:, lo:hi]
        n = min(len(S), len(f0))
        v = (conf[:n] >= CONF_THR) & (f0[:n] > FMIN) & (f0[:n] < FMAX)
        if v.sum() < 100:
            continue
        pk = freqs[lo + np.argmax(S[:n], 1)][v]
        ratio = pk / f0[:n][v]
        h = np.round(ratio)
        ok = (h >= 1) & (np.abs(12 * np.log2(ratio / np.maximum(h, 1))) < 0.5)
        for k in range(1, 12):
            harm_counts[k] += int(np.sum(ok & (h == k)))
        harm_counts[0] += int(np.sum(~ok))  # not on any harmonic (or below F0)
        per_clip.append(float(np.mean(ok & (h == 1))))
        f0s.append(float(np.median(f0[:n][v])))
    tot = harm_counts.sum()
    res = dict(n_hums=len(per_clip), n_frames=int(tot), median_hum_f0_hz=float(np.median(f0s)),
               frac_frames_peak_on_harmonic={("none" if k == 0 else f"h{k}"): float(harm_counts[k] / tot) for k in range(12)},
               per_clip_frac_peak_is_f0=dict(median=float(np.median(per_clip)), q25=float(np.percentile(per_clip, 25)),
                                             q75=float(np.percentile(per_clip, 75))),
               settings=dict(sr=16000, n_fft=N_FFT, win=1024, band_hz=[50, 6000], tol_st=0.5, f0="RMVPE normal speed, conf>=0.3",
                             hum_cache=a.hum_cache))
    json.dump(res, open(os.path.join(a.outdir, "hum_spectral_peak_check.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--mlend-root", required=True); e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=8); e.add_argument("--threads", type=int, default=1)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--model", default="rmvpe.pt"); e.add_argument("--rmvpe-dir", default=".")
    n = sub.add_parser("analyze"); n.add_argument("--mlend-root", default=None); n.add_argument("--cache", required=True)
    n.add_argument("--outdir", required=True); n.add_argument("--peak-feat", default=PEAK_FEAT)
    n.add_argument("--peak-thr", type=float, default=None)
    d = sub.add_parser("downstream"); d.add_argument("--cache", required=True); d.add_argument("--hum-cache", required=True)
    d.add_argument("--outdir", required=True); d.add_argument("--song", default="Potter")
    d.add_argument("--null-songs", default="StarWars,Hakuna")
    d.add_argument("--peak-rules", default="prom:40,prom:50,prom:60,tonal_db:6", help="feature:threshold list")
    d.add_argument("--n-boot", type=int, default=1000)
    d.add_argument("--control-rule", default="tonal_db:6", help="peak voicing rule used in the 2x2 controls")
    s = sub.add_parser("synth"); s.add_argument("--outdir", required=True); s.add_argument("--model", default="rmvpe.pt")
    s.add_argument("--rmvpe-dir", default="."); s.add_argument("--threads", type=int, default=1)
    s.add_argument("--peak-feat", default=PEAK_FEAT); s.add_argument("--peak-thr", type=float, default=PEAK_THR)
    h = sub.add_parser("humpeak"); h.add_argument("--hum-cache", required=True); h.add_argument("--outdir", required=True)
    h.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dict(extract=cmd_extract, analyze=cmd_analyze, downstream=cmd_downstream, synth=cmd_synth,
         humpeak=cmd_humpeak)[a.cmd](a)


if __name__ == "__main__":
    main()
