"""동기 수집기(collect.py)의 세션 폴더를 그대로 가져오기

세션 폴더 (sessions/20261020_231500/ 또는 collections/…)
  audio.wav    마이크 16 kHz            → data/recordings/<참가자>_<세션>.wav
  labels.csv   start_s,end_s,kind        → 같은 이름 .txt (Audacity 라벨, 코골이 학습용)
  emg.npy      턱 EMG µV (1000 SPS)      → data/emg/<참가자>_<세션>_emg.npy (brux_detect.py 용)
  meta.json    샘플링 속도 등             (읽어서 확인만)
  audio_raw.i16 / emg_raw.f32 만 있으면   (비정상 종료) 원본에서 바로 변환

    python tools/import_session.py 세션폴더 --subject P01
    python tools/import_session.py collections --subject P01 --all      # 폴더 안 세션 전부
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snore.dataio import load_wav, write_audacity_labels  # noqa: E402
from tools.import_data import norm_label  # noqa: E402

KIND = {"snore": "snore", "코골이": "snore", "brux": None, "bruxism": None, "clench": None, "이갈이": None}


def one(d, subject, data):
    name = os.path.basename(os.path.normpath(d))
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    afs, efs = int(meta.get("audio_fs", 16000)), int(meta.get("emg_fs", 1000))
    base = f"{subject}_{name}"
    rec = os.path.join(data, "recordings")
    os.makedirs(rec, exist_ok=True)

    wav = os.path.join(d, "audio.wav")
    raw = os.path.join(d, "audio_raw.i16")
    if os.path.exists(wav):
        x = load_wav(wav)
    elif os.path.exists(raw):
        x = np.fromfile(raw, "<i2").astype(np.float32) / 32768
        if afs != 16000:
            from scipy.signal import resample_poly
            x = resample_poly(x, 16000, afs).astype(np.float32)
    else:
        x = None
    if x is not None and len(x):
        wavfile.write(os.path.join(rec, base + ".wav"), 16000, (np.clip(x, -1, 1) * 32767).astype(np.int16))
        print(f"  소리 {len(x) / 16000:6.1f}초 → {os.path.join(rec, base + '.wav')}")

    lp = os.path.join(d, "labels.csv")
    snore_rows, brux_rows = [], []
    if os.path.exists(lp):
        import csv
        for r in csv.DictReader(open(lp, encoding="utf-8-sig")):
            k = (r.get("kind") or r.get("label") or "").strip()
            row = (float(r["start_s"]), float(r["end_s"]), k)
            (brux_rows if KIND.get(k.lower(), "x") is None else snore_rows).append(row)
    if snore_rows and x is not None:
        write_audacity_labels(os.path.join(rec, base + ".txt"), [(s, e, norm_label(k)) for s, e, k in snore_rows])
        print(f"  소리 라벨 {len(snore_rows)}개 → {base}.txt")
    elif x is not None:
        print("  ※ 소리 라벨 없음 → 학습에는 안 쓰이고 detect.py / live.py 시험용으로만 사용 가능")

    emg_d = os.path.join(data, "emg")
    src = os.path.join(d, "emg.npy")
    raw_e = os.path.join(d, "emg_raw.f32")
    e = None
    if os.path.exists(src):
        e = np.load(src)
    elif os.path.exists(raw_e):
        r = np.fromfile(raw_e, "<f4")
        e = r[: len(r) // 2 * 2].reshape(-1, 2)[:, 1]          # [EEG, 턱EMG] 중 턱
    if e is not None and len(e):
        os.makedirs(emg_d, exist_ok=True)
        out = os.path.join(emg_d, f"{base}_emg.npy")
        np.save(out, e.astype(np.float32))
        sat = float(np.mean(np.abs(e) > 2.42 / 6 * 1e6 * 0.98))
        print(f"  EMG {len(e) / efs:6.1f}초 ({efs} SPS) → {out}   포화 {sat * 100:.1f}%"
              + ("  ← 전극 접촉 확인 필요" if sat > 0.05 else ""))
        if brux_rows:
            with open(out.replace(".npy", "_labels.csv"), "w", encoding="utf-8-sig") as f:
                f.write("start_s,end_s,kind\n" + "".join(f"{s},{e2},{k}\n" for s, e2, k in brux_rows))
    for extra in ("events.log", "meta.json"):
        if os.path.exists(os.path.join(d, extra)):
            os.makedirs(os.path.join(data, "sessions_meta"), exist_ok=True)
            shutil.copy(os.path.join(d, extra), os.path.join(data, "sessions_meta", f"{base}_{extra}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--all", action="store_true", help="path 안의 세션 폴더 전부")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    dirs = ([os.path.join(a.path, x) for x in sorted(os.listdir(a.path)) if os.path.isdir(os.path.join(a.path, x))]
            if a.all else [a.path])
    for d in dirs:
        files = set(os.listdir(d))
        if not files & {"audio.wav", "audio_raw.i16", "emg.npy", "emg_raw.f32"}:
            print(f"건너뜀 (측정 파일 없음): {d}")
            continue
        print(f"세션 {d}")
        one(d, a.subject, a.data)


if __name__ == "__main__":
    main()
