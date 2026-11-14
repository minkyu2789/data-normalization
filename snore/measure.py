"""코골이 측정 지표 (한 밤 녹음 → 숫자 요약)

  코골이 시간        : 코골이로 판단한 창 시간 합 (분), 전체 대비 %
  코골이 소리 수     : 한 번 한 번의 코골이 소리 개수, 시간당 개수 (snore index 와 비슷한 개념)
  코골이 구간 수     : 반복으로 확정된 구간 개수, 시간당 개수, 평균 길이
  크기               : 코골이 소리의 방 소음 대비 크기 (dB), 디지털 크기 dBFS
                      ※ 마이크 보정을 안 했으므로 실제 소리 크기(dB SPL)는 아님. 같은 장치끼리 비교용
"""
import numpy as np

from .config import HOP_S
from .features import FEATURE_NAMES

I_REL = FEATURE_NAMES.index("rel_db")
I_PEAK = FEATURE_NAMES.index("peak_db")


def snore_metrics(t, flags, events, episodes, F, floor_db, duration_s):
    hours = max(duration_s / 3600, 1e-9)
    snore_s = float(flags.sum() * HOP_S)
    loud = F[flags, I_PEAK] if flags.any() else np.zeros(0)
    ep_len = [e - s for s, e, _ in episodes]
    return {
        "녹음 길이(분)": round(duration_s / 60, 1),
        "코골이 시간(분)": round(snore_s / 60, 1),
        "코골이 비율(%)": round(100 * snore_s / max(duration_s, 1e-9), 1),
        "코골이 소리 수": len(events),
        "코골이 소리/시간": round(len(events) / hours, 1),
        "코골이 구간 수": len(episodes),
        "코골이 구간/시간": round(len(episodes) / hours, 1),
        "구간 평균 길이(초)": round(float(np.mean(ep_len)), 1) if ep_len else 0.0,
        "코골이 크기 중앙값(방 소음 대비 dB)": round(float(np.median(loud)), 1) if len(loud) else None,
        "코골이 크기 최대(방 소음 대비 dB)": round(float(np.max(loud)), 1) if len(loud) else None,
        "코골이 최대 크기(dBFS)": round(float(np.max(loud) + floor_db), 1) if len(loud) else None,
        "방 바닥소음(dBFS)": round(float(floor_db), 1),
    }


def print_metrics(m, printer=print):
    printer("── 코골이 측정 요약")
    for k, v in m.items():
        printer(f"  {k:28s} {v}")
