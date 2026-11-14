"""데이터 증강 - 적은 데이터로 다양한 상황을 흉내

- 이불 효과   : 고음이 줄고 전체가 작아짐 (저역통과 600–1500Hz, -4~-12dB)
- 배경 소음   : 선풍기·에어컨 같은 분홍 잡음을 SNR 5–25dB로 섞기
- 크기 변화   : ±6dB (마이크 거리 차이)
- 시간 이동   : 창 안에서 소리 위치 이동
"""
import numpy as np
from scipy import signal

from .config import SR


def pink_noise(n, rng):
    w = rng.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    x = np.fft.irfft(f, n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def blanket(x, rng):
    fc = rng.uniform(600, 1500)
    sos = signal.butter(4, fc, btype="low", fs=SR, output="sos")
    return (signal.sosfilt(sos, x) * 10 ** (rng.uniform(-12, -4) / 20)).astype(np.float32)


def add_noise(x, rng, snr_db=None):
    snr_db = rng.uniform(5, 25) if snr_db is None else snr_db
    p = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x + pink_noise(len(x), rng) * p / 10 ** (snr_db / 20)).astype(np.float32)


def gain(x, rng):
    return (x * 10 ** (rng.uniform(-6, 6) / 20)).astype(np.float32)


def shift(x, rng):
    return np.roll(x, rng.integers(0, len(x)))


def augment_recording(x, rng):
    """녹음 단위 증강 (창으로 자르기 전에 적용)"""
    y = x.copy()
    if rng.random() < 0.5:
        y = blanket(y, rng)
    if rng.random() < 0.7:
        y = add_noise(y, rng)
    if rng.random() < 0.5:
        y = gain(y, rng)
    return y
