"""이갈이 예시 근전도 (코드 확인용 합성 데이터)

휴식 잡음 + 60Hz 전원 잡음 + 정해 둔 시각에 근육 활동(위상성 / 긴장성) + 가짜 '삼키기'(한 번짜리) 넣음
정답 구간을 함께 돌려줌
"""
import numpy as np

from .emg import FS


def _burst(n, rng, amp):
    """근육 활동 = 대역 제한 잡음 × 부드러운 포락선"""
    from scipy import signal
    sos = signal.butter(4, [40, 250], btype="band", fs=FS, output="sos")
    v = signal.sosfilt(sos, rng.standard_normal(n))
    v /= v.std() + 1e-9
    env = np.sin(np.linspace(0, np.pi, n)) ** 0.5
    return v * env * amp


def make(minutes=20, seed=3, rest_uv=4.0, mvc_uv=300.0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    t = np.arange(n) / FS
    x = rng.standard_normal(n) * rest_uv + 30 * np.sin(2 * np.pi * 60 * t) + 200 * np.sin(2 * np.pi * 0.2 * t)
    truth = []

    def add(s, d, frac):
        i, k = int(s * FS), int(d * FS)
        x[i:i + k] += _burst(k, rng, frac * mvc_uv)

    for start in rng.choice(np.arange(90, minutes * 60 - 60, 45), size=max(3, minutes // 3), replace=False):
        start = float(start)
        kind = rng.choice(["위상성", "긴장성", "위상성"])
        if kind == "위상성":
            s = start
            for _ in range(rng.integers(4, 8)):
                d = rng.uniform(0.4, 1.2)
                add(s, d, rng.uniform(0.25, 0.6))
                s += d + rng.uniform(0.3, 1.0)
            truth.append((start, s, kind))
        else:
            d = rng.uniform(3, 8)
            add(start, d, rng.uniform(0.2, 0.5))
            truth.append((start, start + d, kind))
    # 한 번짜리 활동(삼키기 등): 이갈이가 아님
    for s in rng.uniform(70, minutes * 60 - 30, size=6):
        if all(abs(s - a) > 20 for a, _, _ in truth):
            add(s, 0.6, 0.4)
    return x, sorted(truth)


def make_calibration(seed=4, rest_uv=4.0, mvc_uv=300.0):
    """꽉 물기 3회 (3초씩, 5초 휴식)"""
    rng = np.random.default_rng(seed)
    n = 25 * FS
    x = rng.standard_normal(n) * rest_uv
    for k in range(3):
        i = (3 + 7 * k) * FS
        x[i:i + 3 * FS] += _burst(3 * FS, rng, mvc_uv * rng.uniform(0.9, 1.0))
    return x


def make_protocol(seed=5, rest_uv=4.0, mvc_uv=300.0):
    """brux_calibrate.py 의 보정 순서대로 만든 가짜 기록 + 표시 [(시작, 끝, 이름)]"""
    from brux_calibrate import PROTOCOL
    rng = np.random.default_rng(seed)
    total = sum(d for _, d, _ in PROTOCOL)
    n = int(total * FS)
    t = np.arange(n) / FS
    x = rng.standard_normal(n) * rest_uv + 20 * np.sin(2 * np.pi * 60 * t)
    marks, s = [], 0.0

    def add(a, d, frac):
        i, k = int(a * FS), int(d * FS)
        x[i:i + k] += _burst(k, rng, frac * mvc_uv)

    for name, dur, _ in PROTOCOL:
        if name.startswith("clench"):
            add(s + 0.2, dur - 0.4, rng.uniform(0.9, 1.0))
        elif name in ("grind", "tap"):
            a = s + 0.5
            while a < s + dur - 1.2:
                d = rng.uniform(0.4, 1.0) if name == "grind" else rng.uniform(0.28, 0.4)
                add(a, d, rng.uniform(0.3, 0.5))
                a += d + rng.uniform(0.3, 0.6)
        elif name == "hold":
            add(s + 0.5, dur - 1.0, 0.2)
        elif name == "swallow":
            for k in range(5):
                add(s + 0.5 + 4 * k, 0.6, 0.35)
        elif name == "talk":
            k = int(dur * FS)
            i = int(s * FS)
            x[i:i + k] += _burst(k, rng, 0.06 * mvc_uv) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 2 * np.arange(k) / FS)))
        elif name == "yawn":
            add(s + 2, 1.0, 0.15)
        if name != "pause":
            marks.append((s, s + dur, name))
        s += dur
    return x, marks
