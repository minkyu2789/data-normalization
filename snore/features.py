"""특징 추출

A. 로그 멜 스펙트로그램 (64 × 101)  → CNN 입력
B. 손으로 만든 특징 49개           → 규칙 기반, 랜덤포레스트 입력
   - MFCC 20개의 평균·표준편차 (40)   : 소리의 음색
   - 대역 에너지 비율 3개            : 저음(50–300) / 중음(300–1000) / 고음(1000–4000)
   - 스펙트럼 중심, 롤오프, 평탄도   : 코골이는 저음·배음 구조, 옷 소리는 고음·잡음형(평탄)
   - 영교차율                        : 잡음일수록 높음
   - 상대 크기(rel_db), 최대 크기     : 바닥소음보다 얼마나 큰지
"""
import numpy as np
from scipy.fft import dct

from .config import HOP_LEN, N_FFT, N_MELS, N_MFCC, SR

FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
    + ["low_ratio", "mid_ratio", "high_ratio", "centroid", "rolloff85", "flatness", "zcr", "rel_db", "peak_db"]
)


def _hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=50, fmax=4000):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb, freqs


_FB, _FREQS = mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def power_spec(wins):
    """(n, samples) → (n, frames, bins)"""
    n, L = wins.shape
    pad = N_FFT // 2
    x = np.pad(wins, ((0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + (x.shape[1] - N_FFT) // HOP_LEN
    idx = np.arange(N_FFT)[None, :] + HOP_LEN * np.arange(n_frames)[:, None]
    frames = x[:, idx] * _WINDOW
    return (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).astype(np.float32)


def logmel(wins):
    p = power_spec(wins)
    m = p @ _FB.T                                   # (n, frames, mels)
    return np.log10(m + 1e-6).transpose(0, 2, 1)    # (n, mels, frames)


def handcrafted(wins, rel_db, peak_db):
    p = power_spec(wins)                             # (n, frames, bins)
    lm = np.log10(p @ _FB.T + 1e-6)
    mf = dct(lm, type=2, axis=-1, norm="ortho")[..., :N_MFCC]
    spec = p.mean(axis=1) + 1e-12                    # (n, bins)
    tot = spec[:, (_FREQS >= 50) & (_FREQS <= 4000)].sum(axis=1)
    band = lambda a, b: spec[:, (_FREQS >= a) & (_FREQS < b)].sum(axis=1) / tot
    centroid = (spec * _FREQS).sum(axis=1) / spec.sum(axis=1)
    cum = np.cumsum(spec, axis=1) / spec.sum(axis=1, keepdims=True)
    rolloff = _FREQS[np.argmax(cum >= 0.85, axis=1)]
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    zcr = (np.abs(np.diff(np.sign(wins), axis=1)) > 0).mean(axis=1)
    return np.column_stack([
        mf.mean(axis=1), mf.std(axis=1),
        band(50, 300), band(300, 1000), band(1000, 4000),
        centroid / 1000, rolloff / 1000, flat, zcr, rel_db, peak_db,
    ]).astype(np.float32)
