from __future__ import annotations

import warnings
import numpy as np
import scipy
import scipy.stats as stats
import joblib
from pathlib import Path


# ─── Feature helpers (match prepare_data.ipynb exactly) ──────────────────────

def _tinn(rri: np.ndarray) -> tuple[float, float]:
    try:
        kernel = stats.gaussian_kde(rri)
        x_axis = np.linspace(rri.min(), rri.max(), len(rri))
        density = kernel.evaluate(x_axis)
        step = x_axis[1] - x_axis[0]

        peak_idx = int(np.argmax(density))
        peak_pos = float(x_axis[peak_idx])
        peak_val = float(density[peak_idx])
        hrv_index = len(rri) / peak_val

        n_axis = x_axis[: peak_idx + 1]
        m_axis = x_axis[peak_idx:]

        err_N: list[tuple[float, float]] = []
        for i in range(len(n_axis) - 1):
            N = n_axis[i]
            slope = peak_val / (peak_pos - N)
            tgt = density[: peak_idx + 1]
            q = np.clip(slope * step * np.arange(-i, -i + peak_idx + 1), 0, None)
            d = tgt - q
            sq = d * d
            err_N.append((float(np.linalg.norm((sq[:-1] + sq[1:]) / 2)), float(N)))

        err_M: list[tuple[float, float]] = []
        for i in range(1, len(m_axis)):
            M = m_axis[i]
            slope = peak_val / (peak_pos - M)
            tgt = density[peak_idx:]
            q = np.clip(slope * step * np.arange(-i, len(tgt) - i), 0, None)
            d = tgt - q
            sq = d * d
            err_M.append((float(np.linalg.norm((sq[:-1] + sq[1:]) / 2)), float(M)))

        if not err_N or not err_M:
            return 0.0, hrv_index

        opt_N = min(err_N, key=lambda x: x[0])[1]
        opt_M = min(err_M, key=lambda x: x[0])[1]
        return float(opt_M - opt_N), hrv_index

    except Exception:
        return 0.0, float(len(rri))


def _nn50(rri: np.ndarray) -> tuple[int, float]:
    if len(rri) < 2:
        return 0, 0.0
    diffs = np.abs(np.diff(rri))
    n = int(np.sum(diffs > 0.05))
    return n, n / len(diffs)


def _freq_features(rri: np.ndarray) -> tuple[float, float, float]:
    mean_rri = float(np.mean(rri))
    fft = scipy.fft.fft(rri - mean_rri)
    half = len(rri) // 2
    freqs = scipy.fft.fftfreq(len(rri), mean_rri)[:half]
    psd = (2.0 / len(fft)) * np.abs(fft)[:half]
    return float(np.mean(freqs)), float(np.std(freqs)), float(np.sum(psd))


def _autonomic_features(rri: np.ndarray) -> tuple[float, float]:
    mean_rri = float(np.mean(rri))
    fft = scipy.fft.fft(rri - mean_rri)
    half = len(rri) // 2
    freqs = scipy.fft.fftfreq(len(rri), mean_rri)[:half]
    psd = (2.0 / len(fft)) * np.abs(fft)[:half]

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    lf = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    return hf, (lf / hf if hf > 0 else 0.0)


def _sd2(rri: np.ndarray) -> float:
    if len(rri) < 2:
        return 0.0
    return float(np.std(rri[:-1] + rri[1:], ddof=1) / np.sqrt(2))


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_features(rri_ms: list[int]) -> np.ndarray | None:
    """Extract 15 HRV features from RRI values in milliseconds.

    Feature order matches prepare_data.ipynb get_data_ecg():
      0  mean_freq   1  std_freq    2  tinn_score  3  hrv_index
      4  nn50        5  pnn50       6  mean_hrv    7  std_hrv
      8  rmssd       9  fmean      10  fstd       11  sum_psd
     12  hf_power   13  lf_hf      14  sd2
    """
    if len(rri_ms) < 10:
        return None

    rri = np.array(rri_ms, dtype=float) / 1000.0  # ms → seconds

    freq = 1.0 / rri
    mean_freq = float(np.mean(freq))
    std_freq  = float(np.std(freq))

    tinn_score, hrv_index = _tinn(rri)
    nn50, pnn50 = _nn50(rri)

    mean_hrv = float(np.mean(rri))
    std_hrv  = float(np.std(rri))
    rmssd    = float(np.sqrt(np.mean(np.diff(rri) ** 2)))

    fmean, fstd, sum_psd = _freq_features(rri)
    hf_power, lf_hf      = _autonomic_features(rri)
    sd2 = _sd2(rri)

    return np.array([
        mean_freq, std_freq, tinn_score, hrv_index,
        nn50,      pnn50,    mean_hrv,   std_hrv,
        rmssd,     fmean,    fstd,       sum_psd,
        hf_power,  lf_hf,    sd2,
    ], dtype=float)


class StressPredictor:
    def __init__(self, model_path: Path | str) -> None:
        bundle = joblib.load(model_path)
        self._model     = bundle["model"]
        self._scaler    = bundle["scaler"]
        self._threshold = float(bundle["threshold"])
        self._baseline: np.ndarray | None = None
        print(f"[Stress] Model loaded (threshold={self._threshold})")

    @property
    def calibrated(self) -> bool:
        return self._baseline is not None

    def set_baseline(self, rri_ms: list[int]) -> bool:
        """Compute and store baseline features from a calm RRI window."""
        features = extract_features(rri_ms)
        if features is None:
            return False
        self._baseline = features
        print(f"[Stress] Baseline set from {len(rri_ms)} beats")
        return True

    def predict(self, rri_ms: list[int]) -> dict | None:
        """Returns {"prob": float, "is_stress": bool} or None if not ready."""
        if self._baseline is None:
            return None

        raw = extract_features(rri_ms)
        if raw is None:
            return None

        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(self._baseline != 0.0, raw / self._baseline, 0.0)

        if np.isinf(norm).any() or np.isnan(norm).any():
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            scaled = self._scaler.transform(norm.reshape(1, -1))
            prob   = float(self._model.predict_proba(scaled)[0][1])
        return {
            "prob":     round(prob, 3),
            "is_stress": bool(prob >= self._threshold),
            "raw":      [round(float(x), 6) for x in raw],
            "norm":     [round(float(x), 4) for x in norm],
        }
