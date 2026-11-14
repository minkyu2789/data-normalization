"""전처리

1. 대역통과 필터 50–4000Hz : 전기 잡음(험), 아주 높은 잡음 제거
2. 바닥소음 추정            : 녹음별 하위 10% 크기 = 그 방의 조용한 상태
3. 1초 창으로 자르기 (0.5초씩 이동)
4. 창마다
   - 상대 크기 rel_db = 창 크기 - 바닥소음  (마이크 거리, 방이 달라도 비교 가능)
   - 크기 정규화 (RMS = 1)                   (사람마다 다른 소리 크기 차이 제거 → 모양만 비교)
"""
import numpy as np
from scipy import signal

from .config import BAND, HOP_S, SR, WIN_S

_SOS = signal.butter(4, BAND, btype="band", fs=SR, output="sos")


def bandpass(x):
    if len(x) < 64:
        return x
    return signal.sosfiltfilt(_SOS, x).astype(np.float32)


def frame_db(x, frame=int(0.05 * SR)):
    n = len(x) // frame
    if n == 0:
        return np.array([-100.0])
    f = x[: n * frame].reshape(n, frame)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)


def noise_floor_db(x):
    return float(np.percentile(frame_db(x), 10))


def segment(x, win_s=WIN_S, hop_s=HOP_S):
    w, h = int(win_s * SR), int(hop_s * SR)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    n = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx], np.arange(n) * hop_s


def prepare(x, floor_db=None):
    """녹음 하나 → (정규화된 창들, 창 시작 시각, 창별 상대 크기 dB, 바닥소음)"""
    x = bandpass(np.asarray(x, dtype=np.float32))
    floor = noise_floor_db(x) if floor_db is None else floor_db
    wins, t = segment(x)
    rms = np.sqrt((wins ** 2).mean(axis=1)) + 1e-9
    rel_db = 20 * np.log10(rms) - floor
    peak_db = np.array([frame_db(w).max() for w in wins]) - floor
    norm = (wins / rms[:, None]).astype(np.float32)
    return norm, t, rel_db.astype(np.float32), peak_db.astype(np.float32), floor
