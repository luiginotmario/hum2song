"""Shared pitch-contour pipeline for the paper figures.

Used by contour_figure.py (the hum/whistle contour figure) and validate_whistle_f0.py (whistle F0
validation and the downstream contour-retrieval check), so both score contours the same way.

  F0 tracks  : RMVPE voicing rule; STFT spectral-peak whistle track with its tonal-ratio gate
  contours   : MIDI semitones, short voiced runs dropped, octave jumps folded, median filter,
               key normalization (minus the clip median), 20 ms frames
  DTW        : the query is linearly resampled to the reference length, then aligned with
               slope-constrained DTW (local tempo ratio 1/2..2). The constraint always applies;
               a pair with no valid path is reported as unaligned (None), never re-aligned with
               unconstrained steps.
"""
from dataclasses import dataclass

import numpy as np

HOP_S = 0.01                 # F0 frame hop (s)
CONF_THR = 0.3               # RMVPE voicing: salience peak >= CONF_THR
FMIN, FMAX = 50.0, 4200.0    # voiced-F0 limits for RMVPE tracks (Hz)
MIN_VOICED_FRAMES = 100      # clips with fewer voiced frames (1 s) are skipped
DS = 2                       # contour decimation for DTW: 10 ms * DS

MIN_RUN = 5                  # voiced runs shorter than this (frames) are dropped
MEDIAN_K = 5                 # median-filter length (frames)
OCTAVE_FOLD_WIN = 51         # running-median window for octave folding (frames)
OCTAVE_FOLD_ST = 9.0         # fold frames further than this from the running median

SR = 32000                   # spectral-peak analysis rate
N_FFT, WIN = 8192, 2048      # 64 ms Hann window, zero-padded to 3.9 Hz bins
BAND = (200.0, 6000.0)       # spectral-peak search band (Hz)
WHISTLE_PEAK_RULE = ("tonal_db", 6.0)  # D-007: voiced when tonal ratio >= 6 dB

DTW_BAND_RADIUS = 0.25       # Sakoe-Chiba band, fraction of the sequence length
DTW_STEPS = np.array([[1, 1], [1, 2], [2, 1]])  # local tempo ratio limited to 1/2..2
DTW_STEP_WEIGHTS = np.array([1.0, 1.5, 1.5])
MAX_TEMPO_RATIO = 2.0        # steepest slope DTW_STEPS allows


@dataclass
class PairScore:
    aligned: np.ndarray      # query mapped onto the reference frame grid
    mae: float               # mean |semitone difference| after alignment
    r: float                 # Pearson r after alignment
    length_ratio: float      # len(query) / len(reference) before length normalization


# ----------------------------------------------------------------------------- F0 tracks
def rmvpe_voiced(f0, conf):
    return (conf >= CONF_THR) & (f0 > FMIN) & (f0 < FMAX)


def peak_voiced(track, rule=WHISTLE_PEAK_RULE):
    """Spectral-peak voicing: track[feature] >= threshold (prom = peak dB over the in-band median;
    tonal_db = energy within +-50 cents of the peak over the rest of the band)."""
    feature, threshold = rule
    return track[feature] >= threshold


def parabolic_peak(level_db, k):
    """Sub-bin offset of the peak at bin k per frame, parabola through 3 dB values."""
    rows = np.arange(len(k))
    a, b, c = level_db[rows, k - 1], level_db[rows, k], level_db[rows, k + 1]
    den = a - 2 * b + c
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return np.clip(delta, -0.5, 0.5), b


def nearest_bin(freq, df, n_bins):
    return np.clip(np.round(freq / df).astype(int), 0, n_bins - 1)


def level_at(S, rows, bins):
    return 20 * np.log10(S[rows, bins] + 1e-10)


def spectral_track(y, sr=SR):
    """Per-frame spectral peak (Hz, parabolic-interpolated), peak prominence (dB over in-band
    median), tonal energy ratio (dB, energy within +-50 cents of the peak vs the rest of the band),
    in-band spectral flatness, 2nd-harmonic and sub-harmonic (f/2) level relative to the peak (dB),
    frame level (dBFS)."""
    import librosa
    hop = int(sr * HOP_S)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop, win_length=WIN, window="hann",
                            center=True)).T
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    lo, hi = np.searchsorted(freqs, BAND[0]), np.searchsorted(freqs, BAND[1])
    band_mag = S[:, lo:hi] + 1e-10
    level_db = 20 * np.log10(band_mag)
    k = np.clip(np.argmax(band_mag, axis=1), 1, band_mag.shape[1] - 2)
    delta, peak_db = parabolic_peak(level_db, k)
    df = freqs[1] - freqs[0]
    peak_f0 = freqs[lo] + (k + delta) * df
    power = band_mag ** 2
    near_peak = np.abs(12 * np.log2(freqs[lo:hi][None, :] / peak_f0[:, None])) <= 0.5
    tonal = (power * near_peak).sum(1)
    rest = np.maximum(power.sum(1) - tonal, 1e-20)
    rows = np.arange(len(k))
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=hop, center=True)[0]
    track = dict(peak_f0=peak_f0, prom=peak_db - np.median(level_db, axis=1),
                 tonal_db=10 * np.log10(tonal / rest),
                 flat=np.exp(np.mean(np.log(power), 1)) / np.mean(power, 1),
                 h2_db=level_at(S, rows, nearest_bin(2 * peak_f0, df, S.shape[1])) - peak_db,
                 sub_db=level_at(S, rows, nearest_bin(0.5 * peak_f0, df, S.shape[1])) - peak_db,
                 level_db=20 * np.log10(rms + 1e-10))
    n = min(len(v) for v in track.values())
    return {name: v[:n] for name, v in track.items()}


# ----------------------------------------------------------------------------- contours
def drop_short_runs(mask, min_run=MIN_RUN):
    """Unset voiced runs shorter than min_run frames."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    out = mask.copy()
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        out[start:end] = out[start:end] & (end - start >= min_run)
    return out


def fold_octave_jumps(seq):
    """Move frames further than OCTAVE_FOLD_ST from a 0.5 s running median by whole octaves
    toward it."""
    from scipy.signal import medfilt
    running = medfilt(seq, OCTAVE_FOLD_WIN)
    octaves = np.round((running - seq) / 12.0)
    return seq + 12.0 * octaves * (np.abs(running - seq) > OCTAVE_FOLD_ST)


def clean_semitones(f0, voiced):
    """MIDI semitones on voiced frames (NaN elsewhere): short runs dropped, octave jumps folded,
    median-filtered."""
    from scipy.signal import medfilt
    keep = drop_short_runs(voiced)
    st = np.full(len(f0), np.nan)
    st[keep] = 69 + 12 * np.log2(f0[keep] / 440.0)
    idx = np.flatnonzero(keep)
    if len(idx) >= OCTAVE_FOLD_WIN:
        st[idx] = fold_octave_jumps(st[idx])
    if len(idx) >= MEDIAN_K:
        st[idx] = medfilt(st[idx], MEDIAN_K)
    return st


def voiced_seq(st):
    idx = np.flatnonzero(~np.isnan(st))
    return idx, st[idx]


def downsample(x, k=DS):
    n = len(x) // k
    return x[: n * k].reshape(n, k).mean(1) if n > 0 else x


def key_normalized(f0, voiced):
    """Cleaned semitone track, its voiced median and the key-normalized DTW sequence, or None when
    fewer than MIN_VOICED_FRAMES frames are voiced."""
    st = clean_semitones(np.asarray(f0, float), voiced)
    _, seq = voiced_seq(st)
    if len(seq) < MIN_VOICED_FRAMES:
        return None
    median = float(np.median(seq))
    return st, median, downsample(seq - median)


# ----------------------------------------------------------------------------- DTW
def resample_to_length(x, n):
    """Linear resampling of x to n points (endpoints kept)."""
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def slope_constrained_path(ref, qry):
    """Warping path (ref index, qry index), start to end, with steps (1,1),(1,2),(2,1) inside the
    Sakoe-Chiba band; None when no such path exists."""
    import librosa
    cost = np.abs(ref[:, None] - qry[None, :])
    try:
        acc, path = librosa.sequence.dtw(C=cost, subseq=False, global_constraints=True,
                                         band_rad=DTW_BAND_RADIUS, step_sizes_sigma=DTW_STEPS,
                                         weights_add=np.zeros(len(DTW_STEPS)),
                                         weights_mul=DTW_STEP_WEIGHTS)
    except librosa.util.exceptions.ParameterError:
        return None
    return path[::-1] if np.isfinite(acc[-1, -1]) else None


def project_onto_reference(n_ref, qry, path):
    """Mean of the query frames matched to each reference frame; unmatched frames interpolated."""
    total = np.bincount(path[:, 0], weights=qry[path[:, 1]], minlength=n_ref)
    count = np.bincount(path[:, 0], minlength=n_ref)
    matched = count > 0
    out = np.zeros(n_ref)
    out[matched] = total[matched] / count[matched]
    out[~matched] = np.interp(np.flatnonzero(~matched), np.flatnonzero(matched), out[matched])
    return out


def score_pair(ref, qry):
    """Align qry to ref (both key-normalized semitone sequences) and score them. The query is first
    resampled to the reference length (global tempo), so the slope-constrained path always exists
    unless the band excludes it; returns None for a pair that cannot be aligned."""
    qry_norm = resample_to_length(qry, len(ref))
    path = slope_constrained_path(ref, qry_norm)
    if path is None:
        return None
    aligned = project_onto_reference(len(ref), qry_norm, path)
    return PairScore(aligned=aligned, mae=float(np.mean(np.abs(aligned - ref))),
                     r=float(np.corrcoef(aligned, ref)[0, 1]), length_ratio=len(qry) / len(ref))


def outside_tempo_range(length_ratio):
    """True when the raw lengths differ by more than MAX_TEMPO_RATIO (the pairs the old code
    silently aligned with unconstrained steps)."""
    return not (1.0 / MAX_TEMPO_RATIO <= length_ratio <= MAX_TEMPO_RATIO)
