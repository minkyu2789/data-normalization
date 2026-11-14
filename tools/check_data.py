"""데이터셋 점검: 학습 전에 문제 있는 녹음을 찾는다

    python tools/check_data.py --data data

확인 항목
  - 참가자 × 라벨별 녹음 시간 (부족한 칸 표시)
  - 클리핑(소리가 최대치에 닿음) 비율    → 1% 넘으면 마이크를 멀리 / 게인 낮추기
  - 바닥소음, 소리 크기                  → 너무 조용하면 녹음이 안 된 것
  - 코골이 녹음 안에서 실제 소리가 있는 창 비율 → 너무 낮으면 라벨 범위가 넓은 것
  - 라벨 없는 긴 녹음
결과: results/data_check.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snore.config import LABEL_KO, SR, is_snore  # noqa: E402
from snore.dataio import load_dataset  # noqa: E402
from snore.pipeline import SOUND_MIN_DB  # noqa: E402
from snore.preprocess import bandpass, noise_floor_db, prepare  # noqa: E402

MIN_SEC = 60          # 참가자 × 라벨마다 최소 이만큼은 있으면 좋음


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    clips = load_dataset(a.data)
    if not clips:
        raise SystemExit(f"{a.data} 에 녹음이 없어요")

    rows = []
    for c in clips:
        x = c.audio
        clip_ratio = float(np.mean(np.abs(x) > 0.99))
        xb = bandpass(x)
        floor = noise_floor_db(xb)
        _, _, rel, peak, _ = prepare(x, floor_db=floor)
        loud = float(np.mean(peak >= SOUND_MIN_DB))
        warn = []
        if clip_ratio > 0.01:
            warn.append("클리핑")
        if floor < -85:
            warn.append("너무 조용함(녹음 확인)")
        if is_snore(c.label) and loud < 0.15:
            warn.append("코골이 소리 적음(라벨 범위 확인)")
        if len(x) < SR * 2:
            warn.append("2초 미만")
        rows.append({"참가자": c.subject, "라벨": LABEL_KO.get(c.label, c.label), "파일": os.path.basename(c.source),
                     "시작(초)": round(c.offset, 1), "길이(초)": round(len(x) / SR, 1),
                     "클리핑%": round(clip_ratio * 100, 2), "바닥소음 dBFS": round(floor, 1),
                     "평균 상대크기 dB": round(float(np.median(rel)), 1), "소리 창 비율": round(loud, 2),
                     "경고": ", ".join(warn)})
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "data_check.csv"), index=False, encoding="utf-8-sig")

    pv = df.pivot_table(index="참가자", columns="라벨", values="길이(초)", aggfunc="sum", fill_value=0)
    print("── 참가자 × 라벨 녹음 시간 (초)   * = 60초 미만")
    print(pv.map(lambda v: f"{v:.0f}{'*' if v < MIN_SEC else ''}").to_string())
    bad = df[df["경고"] != ""]
    print(f"\n── 경고 {len(bad)}건 / 전체 {len(df)}개")
    if len(bad):
        print(bad[["참가자", "라벨", "파일", "시작(초)", "경고"]].to_string(index=False))
    n_sub = df["참가자"].nunique()
    print(f"\n참가자 {n_sub}명" + ("  ※ 교차검증에는 최소 3명이 필요해요" if n_sub < 3 else ""))

    rec = os.path.join(a.data, "recordings")
    if os.path.isdir(rec):
        nolab = [f for f in os.listdir(rec) if f.endswith(".wav") and not os.path.exists(os.path.join(rec, f[:-4] + ".txt"))]
        if nolab:
            print("라벨(.txt) 없는 긴 녹음:", ", ".join(nolab))
    print(f"\n저장: {os.path.join(a.out, 'data_check.csv')}")


if __name__ == "__main__":
    main()
