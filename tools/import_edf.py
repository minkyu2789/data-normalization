"""수면다원검사 EDF에서 근전도 채널 꺼내기 (공개 데이터: PhysioNet CAP Sleep Database 의 brux1, brux2)

1. https://physionet.org/content/capslpdb/1.0.0/ 에서 brux1.edf, brux2.edf 받기 (로그인 없이, 각 200 MB 이상)
2. python tools/import_edf.py brux1.edf --list           # 채널 목록 보기
3. python tools/import_edf.py brux1.edf                  # 턱(chin) 근전도 자동 선택 → data/emg/CAP_brux1_emg.npy
   python tools/import_edf.py brux1.edf --channel "EMG1-EMG2"
4. python brux_detect.py --emg data/emg/CAP_brux1_emg.npy --fs <출력된 Hz> --warmup 0

주의: CAP 데이터는 이갈이 '환자'의 밤 기록이지만 이갈이 구간 정답 표시는 없음(수면 단계만 있음).
     → 우리 탐지기가 실제 근전도에서 어떻게 동작하는지 '보는' 용도. 정확도(F1) 계산은 불가.
     또 우리 장치는 관자놀이/교근, CAP 은 턱 밑(chin) 근전도라 위치가 다름.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brux.edf import channel_list, read_edf  # noqa: E402

KEYS = ("chin", "ment", "sub", "emg1", "emg", "mass", "tempor")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--channel")
    ap.add_argument("--name")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    chans = channel_list(a.edf)
    if a.list:
        for lab, fs, unit in chans:
            print(f"  {lab:20s} {fs:7.1f} Hz  {unit}")
        return
    names = [c[0] for c in chans]
    pick = a.channel
    if not pick:
        for k in KEYS:
            hit = [n for n in names if k in n.lower() and "tib" not in n.lower() and "leg" not in n.lower()]
            if hit:
                pick = hit[0]
                break
    if not pick or pick not in names:
        raise SystemExit("근전도 채널을 못 찾았어요. --list 로 보고 --channel 로 지정하세요")
    x, fs = read_edf(a.edf, pick=[pick])[pick]
    name = a.name or f"CAP_{os.path.splitext(os.path.basename(a.edf))[0]}"
    d = os.path.join(a.data, "emg")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{name}_emg.npy")
    np.save(out, x)
    json.dump({"source": os.path.basename(a.edf), "channel": pick, "fs": fs, "unit": "uV"},
              open(out.replace(".npy", ".json"), "w"), ensure_ascii=False)
    print(f"채널 '{pick}'  {fs:.0f} Hz  {len(x) / fs / 3600:.2f}시간 → {out}")
    if fs < 400:
        print("※ 샘플링이 낮아서 필터 상한이 자동으로 낮아짐 (근전도 고주파 일부 손실)")
    print(f"다음: python brux_detect.py --emg {out} --fs {fs:.0f} --warmup 0")


if __name__ == "__main__":
    main()
