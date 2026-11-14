"""예시 데이터 만들기 (실제 녹음이 없을 때 코드 동작 확인용)

참가자마다 코골이 높이(기본 주파수), 호흡 주기, 소리 크기, 방 소음이 다르게 만들어서
'사람마다 다르다'는 상황을 흉내 냄. 실제 성능은 반드시 실제 녹음으로 다시 확인해야 함.
"""
import os

import numpy as np
from scipy import signal

from .config import SR
from .dataio import save_wav, write_audacity_labels

LABELS = ["snore", "snore_covered", "cloth", "movement", "speech", "quiet"]


def _env(n, a, b, rng):
    e = np.zeros(n)
    i, j = int(a * SR), int(b * SR)
    j = min(j, n)
    if j > i:
        e[i:j] = np.sin(np.linspace(0, np.pi, j - i)) ** rng.uniform(0.6, 1.5)
    return e


def room(n, rng, level):
    w = rng.standard_normal(n)
    pink = signal.lfilter([0.049922, -0.095993, 0.050612, -0.004408], [1, -2.494956, 2.017265, -0.522189], w)
    hum = 0.3 * np.sin(2 * np.pi * rng.choice([50, 60, 120]) * np.arange(n) / SR)
    x = pink / (np.std(pink) + 1e-9) + hum
    return x * level


def snore(dur, p, rng, covered=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    s = rng.uniform(0, 1)
    while s < dur - 1:
        L = rng.uniform(0.7, 1.4)
        f0 = p["f0"] * rng.uniform(0.9, 1.1)
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / SR
        src = signal.sawtooth(ph, 0.3) + 0.4 * rng.standard_normal(n)       # 연구개 떨림 + 바람 소리
        b, a = signal.iirpeak(p["formant"], 2.0, SR)
        tone = signal.lfilter(b, a, src) + 0.3 * src
        out += tone * _env(n, s, s + L, rng) * rng.uniform(0.7, 1.0)
        s += p["breath"] * rng.uniform(0.85, 1.15)
    out *= p["loud"]
    if covered:
        sos = signal.butter(4, rng.uniform(600, 1100), btype="low", fs=SR, output="sos")
        out = signal.sosfilt(sos, out) * 0.35
    return out


def cloth(dur, rng, p):
    n = int(dur * SR)
    out = np.zeros(n)
    s = rng.uniform(0, 0.5)
    while s < dur - 0.3:
        L = rng.uniform(0.2, 2.0)
        nz = rng.standard_normal(n)
        sos = signal.butter(2, [rng.uniform(800, 1500), rng.uniform(4500, 7000)], btype="band", fs=SR, output="sos")
        crackle = signal.sosfilt(sos, nz) * (1 + 2 * (rng.random(n) < 0.02))
        out += crackle * _env(n, s, s + L, rng) * rng.uniform(0.3, 1.0)
        s += L + rng.uniform(0.3, 4.0)
    return out * p["loud"] * 0.8


def movement(dur, rng, p):
    n = int(dur * SR)
    out = cloth(dur, rng, p) * 0.6
    for s in rng.uniform(0, dur - 0.5, int(dur / 3)):
        i = int(s * SR)
        k = np.arange(int(0.25 * SR))
        thump = np.sin(2 * np.pi * rng.uniform(40, 90) * k / SR) * np.exp(-k / (0.04 * SR))
        out[i:i + len(k)] += thump[: n - i] * p["loud"] * rng.uniform(0.8, 2.0)
    return out


def speech(dur, rng, p):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    s = rng.uniform(0, 1)
    while s < dur - 1:
        L = rng.uniform(0.8, 2.5)
        f0 = p["f0_speech"] * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(1, 3) * t))
        src = signal.sawtooth(2 * np.pi * np.cumsum(f0) / SR, 0.1)
        y = np.zeros(n)
        for fm in (rng.uniform(400, 800), rng.uniform(1000, 1800), rng.uniform(2200, 3000)):
            b, a = signal.iirpeak(fm, 4.0, SR)
            y += signal.lfilter(b, a, src)
        syl = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 5) * t) ** 2       # 음절 리듬
        out += y * syl * _env(n, s, s + L, rng)
        s += L + rng.uniform(1, 5)
    return out * p["loud"] * 0.5


def subject_params(i, rng):
    return {
        "f0": rng.uniform(55, 140), "formant": rng.uniform(250, 600), "breath": rng.uniform(3.2, 5.0),
        "loud": rng.uniform(0.05, 0.2), "room": rng.uniform(0.002, 0.008), "f0_speech": rng.uniform(100, 220),
    }


def make_clip(label, dur, p, rng):
    n = int(dur * SR)
    base = room(n, rng, p["room"])
    gen = {
        "snore": lambda: snore(dur, p, rng),
        "snore_covered": lambda: snore(dur, p, rng, covered=True),
        "cloth": lambda: cloth(dur, rng, p),
        "movement": lambda: movement(dur, rng, p),
        "speech": lambda: speech(dur, rng, p),
        "quiet": lambda: np.zeros(n),
    }[label]
    return (base + gen()).astype(np.float32)


def make_demo(root, n_subjects=6, sec_per_label=30, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n_subjects):
        subj = f"P{i + 1:02d}"
        p = subject_params(i, rng)
        for lab in LABELS:
            save_wav(os.path.join(root, "subjects", subj, lab, "001.wav"), make_clip(lab, sec_per_label, p, rng))

    # 긴 녹음 + Audacity 라벨 (실시간 탐지 시연용, 학습에는 쓰지 않도록 별도 폴더)
    p = subject_params(99, rng)
    plan = [("quiet", 40), ("snore", 60), ("cloth", 15), ("snore_covered", 50), ("speech", 20),
            ("quiet", 30), ("movement", 20), ("snore", 70), ("cloth", 10), ("quiet", 25), ("snore", 40)]
    parts, rows, t = [], [], 0.0
    for lab, d in plan:
        parts.append(make_clip(lab, d, p, rng))
        rows.append((t, t + d, lab))
        t += d
    night = np.concatenate(parts)
    save_wav(os.path.join(root, "test", "P99_night.wav"), night)
    write_audacity_labels(os.path.join(root, "test", "P99_night.txt"), rows)
