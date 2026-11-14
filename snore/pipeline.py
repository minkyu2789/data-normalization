"""녹음 → 창별 특징 데이터셋

라벨 정제: 코골이 녹음은 '코골이 소리 – 숨 쉬는 조용한 틈 – 코골이 소리'가 반복되는데,
녹음 전체에 '코골이' 라벨을 붙이면 조용한 틈까지 코골이로 학습됨 → 일관성이 깨짐.
그래서 코골이 녹음에서는 바닥소음보다 6dB 이상 큰 창만 '코골이'로 사용."""
from dataclasses import dataclass

import numpy as np

from .augment import augment_recording
from .config import PER_CLIP_FLOOR_PREFIX, is_snore
from .dataio import load_wav
from .features import handcrafted, logmel
from .preprocess import bandpass, noise_floor_db, prepare


@dataclass
class WindowSet:
    F: np.ndarray        # (n, 49) 손으로 만든 특징
    M: np.ndarray        # (n, 64, 101) 로그 멜 (CNN용, 안 쓰면 비어 있음)
    y: np.ndarray        # 1 = 코골이
    subject: np.ndarray
    label: np.ndarray
    clip: np.ndarray     # 원본 클립 번호
    t: np.ndarray        # 클립 안 시각
    aug: np.ndarray      # 0 = 원본, 1 이상 = 증강본

    def take(self, idx):
        return WindowSet(*(a[idx] if len(a) else a for a in
                           (self.F, self.M, self.y, self.subject, self.label, self.clip, self.t, self.aug)))


def subject_floors(clips):
    """참가자별 바닥소음: 'quiet' 녹음이 있으면 그 값, 없으면 그 사람 녹음 중 가장 조용한 값.
    긴 녹음(Audacity 라벨)에서 잘라낸 구간은 원본 녹음 전체로 계산."""
    floors, rec_cache = {}, {}
    for c in clips:
        if c.offset or c.source.endswith(".wav") and "recordings" in c.source.replace("\\", "/"):
            if c.source not in rec_cache:
                rec_cache[c.source] = noise_floor_db(bandpass(load_wav(c.source)))
    for s in {c.subject for c in clips}:
        mine = [c for c in clips if c.subject == s and c.source not in rec_cache]
        quiet = [noise_floor_db(bandpass(c.audio)) for c in mine if c.label in ("quiet", "조용함")]
        allf = [noise_floor_db(bandpass(c.audio)) for c in mine]
        floors[s] = float(np.median(quiet)) if quiet else (float(np.min(allf)) if allf else None)
        if str(s).startswith(PER_CLIP_FLOOR_PREFIX):
            floors[s] = None          # 클립마다 따로
    return floors, rec_cache


QUIET_LABELS = {"quiet", "조용함"}
SOUND_MIN_DB = 6.0   # 코골이 녹음 안에서 이보다 조용한 창은 '숨 사이 쉬는 구간'이라 학습에서 뺌


def build(clips, n_aug=0, with_mel=False, seed=0):
    rng = np.random.default_rng(seed)
    floors, rec_floor = subject_floors(clips)
    cols = {k: [] for k in ("F", "M", "y", "subject", "label", "clip", "t", "aug")}
    for ci, c in enumerate(clips):
        floor = rec_floor.get(c.source, floors.get(c.subject))
        versions = [c.audio] + [augment_recording(c.audio, rng) for _ in range(n_aug)]
        for k, x in enumerate(versions):
            norm, t, rel, peak, _ = prepare(x, floor_db=floor)
            if is_snore(c.label) and floor is not None:   # 공개 데이터 클립은 이미 무음이 잘려 있음
                # 라벨 정제: '코골이' 녹음에도 숨 사이 조용한 구간이 섞여 있으므로 소리가 있는 창만 사용
                keep = peak >= SOUND_MIN_DB
                norm, t, rel, peak = norm[keep], t[keep], rel[keep], peak[keep]
            n = len(t)
            if n == 0:
                continue
            labs = np.full(n, c.label, dtype=object)
            if floor is not None and not is_snore(c.label) and c.label not in QUIET_LABELS:
                # 옷·뒤척임·말소리 녹음 안의 조용한 틈은 '조용함'으로 (소리 종류 분류가 일관되도록)
                labs[peak < SOUND_MIN_DB] = "quiet"

            cols["F"].append(handcrafted(norm, rel, peak))
            if with_mel:
                cols["M"].append(logmel(norm).astype(np.float16))
            cols["y"].append(np.full(n, int(is_snore(c.label))))
            cols["subject"].append(np.full(n, c.subject, dtype=object))
            cols["label"].append(labs)
            cols["clip"].append(np.full(n, ci))
            cols["t"].append(t + c.offset)
            cols["aug"].append(np.full(n, k))
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    return WindowSet(cat(cols["F"]), cat(cols["M"]), cat(cols["y"]), cat(cols["subject"]),
                     cat(cols["label"]), cat(cols["clip"]), cat(cols["t"]), cat(cols["aug"]))


def windows_from_audio(x, floor_db=None, with_mel=False):
    """긴 녹음 하나 (실시간 탐지용)"""
    norm, t, rel, peak, floor = prepare(x, floor_db=floor_db)
    F = handcrafted(norm, rel, peak)
    M = logmel(norm).astype(np.float16) if with_mel else np.zeros(0)
    return F, M, t, floor
