import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats


# =========================
# TINN
# =========================

def TINN(x):

    kernel = stats.gaussian_kde(x)

    x_axis = np.linspace(
        np.min(x),
        np.max(x),
        len(x)
    )

    density_values = kernel.evaluate(x_axis)

    step_size = x_axis[1] - x_axis[0]

    max_idx = np.argmax(density_values)
    max_pos = x_axis[max_idx]
    max_value = np.amax(density_values)

    n_axis = x_axis[:max_idx + 1]
    m_axis = x_axis[max_idx:]

    hrv_index = len(x) / max_value

    err_N = []
    err_M = []

    for i in range(0, len(n_axis) - 1):

        N = n_axis[i]

        slope = max_value / (max_pos - N)

        target_density = density_values[:max_idx + 1]

        q = np.clip(
            slope * step_size *
            np.arange(-i, -i + max_idx + 1),
            0,
            None
        )

        diff = target_density - q
        squared_error = diff * diff

        err1 = np.delete(squared_error, -1)
        err2 = np.delete(squared_error, 0)

        trapezoidal_error = (err1 + err2) / 2

        total_error = np.linalg.norm(
            trapezoidal_error
        )

        err_N.append((total_error, N))

    for i in range(1, len(m_axis)):

        M = m_axis[i]

        slope = max_value / (max_pos - M)

        target_density = density_values[max_idx:]

        q = np.clip(
            slope * step_size *
            np.arange(
                -i,
                len(target_density) - i
            ),
            0,
            None
        )

        diff = target_density - q
        squared_error = diff * diff

        err1 = np.delete(squared_error, -1)
        err2 = np.delete(squared_error, 0)

        trapezoidal_error = (err1 + err2) / 2

        total_error = np.linalg.norm(
            trapezoidal_error
        )

        err_M.append((total_error, M))

    return err_N, err_M, hrv_index


def best_TINN(x):

    err_N, err_M, hrv_index = TINN(x)

    min_n_idx = np.argmin(
        np.array(err_N, dtype=object)[:, 0]
    )

    min_m_idx = np.argmin(
        np.array(err_M, dtype=object)[:, 0]
    )

    optimal_N = err_N[min_n_idx][1]
    optimal_M = err_M[min_m_idx][1]

    tinn_score = optimal_M - optimal_N

    return tinn_score, hrv_index


# =========================
# NN50
# =========================

def compute_nn50(x):

    if len(x) < 2:
        return 0, 0

    diffs = np.abs(np.diff(x))

    nn50 = int(
        np.sum(diffs > 0.05)
    )

    pnn50 = nn50 / len(diffs)

    return nn50, pnn50


# =========================
# Frequency
# =========================

def get_freq_features(x):

    mean_rri = np.mean(x)

    fft_values = scipy.fft.fft(
        x - mean_rri
    )

    half_len = len(x) // 2

    frequencies = scipy.fft.fftfreq(
        len(x),
        mean_rri
    )[:half_len]

    psd = (
        2.0 / len(fft_values)
    ) * np.abs(fft_values)[:half_len]

    return (
        np.mean(frequencies),
        np.std(frequencies),
        np.sum(psd)
    )


# =========================
# HF LF/HF
# =========================

def get_autonomic_features(x):

    mean_rri = np.mean(x)

    fft_values = scipy.fft.fft(
        x - mean_rri
    )

    half_len = len(x) // 2

    frequencies = scipy.fft.fftfreq(
        len(x),
        mean_rri
    )[:half_len]

    psd = (
        2.0 / len(fft_values)
    ) * np.abs(fft_values)[:half_len]

    lf_mask = (
        (frequencies >= 0.04)
        &
        (frequencies <= 0.15)
    )

    hf_mask = (
        (frequencies >= 0.15)
        &
        (frequencies <= 0.40)
    )

    lf_power = (
        np.trapz(
            psd[lf_mask],
            frequencies[lf_mask]
        )
        if np.any(lf_mask)
        else 0
    )

    hf_power = (
        np.trapz(
            psd[hf_mask],
            frequencies[hf_mask]
        )
        if np.any(hf_mask)
        else 0
    )

    lf_hf = (
        lf_power / hf_power
        if hf_power > 0
        else 0
    )

    return hf_power, lf_hf


# =========================
# SD2
# =========================

def compute_sd2(x):

    if len(x) < 2:
        return 0

    x1 = x[:-1]
    x2 = x[1:]

    return (
        np.std(
            x1 + x2,
            ddof=1
        )
        / np.sqrt(2)
    )


# =========================
# 15 features
# =========================

def extract_features(rri_ms):

    rri = np.array(rri_ms) / 1000.0

    freq = 1.0 / rri

    mean_freq = np.mean(freq)
    std_freq = np.std(freq)

    tinn_score, hrv_index = best_TINN(rri)

    nn50, pnn50 = compute_nn50(rri)

    mean_hrv = np.mean(rri)
    std_hrv = np.std(rri)
    rms_hrv = np.sqrt(
        np.mean(rri ** 2)
    )

    fmean, fstd, sum_psd = \
        get_freq_features(rri)

    hf_power, lf_hf = \
        get_autonomic_features(rri)

    sd2 = compute_sd2(rri)

    return [
        mean_freq,
        std_freq,
        tinn_score,
        hrv_index,
        nn50,
        pnn50,
        mean_hrv,
        std_hrv,
        rms_hrv,
        fmean,
        fstd,
        sum_psd,
        hf_power,
        lf_hf,
        sd2,
    ]


# =========================
# Main
# =========================

WINDOW_SEC = 20

csv_path = "polar_rri_20260623_001651.csv"

df = pd.read_csv(csv_path)

start_ts = df["timestamp"].min()

df["window"] = (
    (df["timestamp"] - start_ts)
    // WINDOW_SEC
).astype(int)

rows = []

for window_id, g in df.groupby("window"):

    rri = g["rri_ms"].values

    if len(rri) < 10:
        continue

    features = extract_features(rri)

    rows.append(
        [window_id] + features
    )

cols = [
    "window",
    "MeanFrequency",
    "StdFrequency",
    "TINN",
    "HRVIndex",
    "NN50",
    "pNN50",
    "MeanHRV",
    "StdHRV",
    "RMSHRV",
    "FFTMean",
    "FFTStd",
    "PSD",
    "HF",
    "LFHF",
    "SD2",
]

result = pd.DataFrame(
    rows,
    columns=cols
)

print(result.head())

result.to_csv(
    "polar_features.csv",
    index=False
)

print(
    "\nSaved: polar_features.csv"
)