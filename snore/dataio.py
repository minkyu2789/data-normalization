"""데이터 불러오기

지원하는 폴더 구조 (둘 다 섞어서 써도 됨)

A. 소리 종류별 파일
   data/subjects/<참가자>/<라벨>/*.wav
   예) data/subjects/P01/snore/001.wav

B. 길게 녹음한 파일 + Audacity 라벨
   data/recordings/<참가자>_<아무이름>.wav
   data/recordings/<참가자>_<아무이름>.txt   (Audacity: 시작 \\t 끝 \\t 라벨)
   라벨이 없는 구간은 학습에 쓰지 않음
"""
import os
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import SR


@dataclass
class Clip:
    subject: str
    label: str
    audio: np.ndarray        # float32, -1~1, SR
    source: str
    offset: float = 0.0      # 원본 파일 안에서 시작 시각 (초)


def load_wav(path, sr=SR):
    file_sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128) / 128
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if file_sr != sr:
        g = np.gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    return x


def save_wav(path, x, sr=SR):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def read_audacity_labels(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0].replace(".", "", 1).isdigit():
                out.append((float(p[0]), float(p[1]), p[2].strip()))
    return out


def write_audacity_labels(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, lab in rows:
            f.write(f"{s:.3f}\t{e:.3f}\t{lab}\n")


def load_dataset(root):
    clips = []
    sub_dir = os.path.join(root, "subjects")
    if os.path.isdir(sub_dir):
        for subj in sorted(os.listdir(sub_dir)):
            for lab in sorted(os.listdir(os.path.join(sub_dir, subj))):
                d = os.path.join(sub_dir, subj, lab)
                if not os.path.isdir(d):
                    continue
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith(".wav"):
                        p = os.path.join(d, fn)
                        clips.append(Clip(subj, lab, load_wav(p), p))

    rec_dir = os.path.join(root, "recordings")
    if os.path.isdir(rec_dir):
        for fn in sorted(os.listdir(rec_dir)):
            if not fn.lower().endswith(".wav"):
                continue
            p = os.path.join(rec_dir, fn)
            lp = p[:-4] + ".txt"
            if not os.path.exists(lp):
                continue
            subj = fn.split("_")[0]
            x = load_wav(p)
            for s, e, lab in read_audacity_labels(lp):
                a, b = int(s * SR), int(e * SR)
                if b - a >= SR:                       # 1초 미만 구간은 버림
                    clips.append(Clip(subj, lab, x[a:b], p, s))
    return clips


def summary_table(clips):
    """참가자 × 라벨별 녹음 시간(초)"""
    subs = sorted({c.subject for c in clips})
    labs = sorted({c.label for c in clips})
    tab = {s: {l: 0.0 for l in labs} for s in subs}
    for c in clips:
        tab[c.subject][c.label] += len(c.audio) / SR
    return subs, labs, tab
