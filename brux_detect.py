"""이갈이 측정: 전처리 → 정규화 → 버스트/에피소드 → 지표 → 자극 요청 판단

  python brux_detect.py --demo                                         가짜 근전도 20분으로 전체 과정
  python brux_detect.py --emg 세션폴더/emg.npy --calib results/brux_calib_P01.json
  python brux_detect.py --emg data/emg/P01_night_emg.npy               보정 없이 (배경 2배 기준만)
  python brux_detect.py --emg 기록_emg.csv --column ch1_uV --fs 1000

입력: .npy (µV 배열, 동기 장치 emg.npy 그대로) 또는 CSV (시간 열 + µV 열)
      옆에 같은 이름 .json 에 fs 가 있으면 자동으로 읽음 (import_edf.py 결과)
결과: results/brux_<이름>.png, _episodes.csv, _summary.csv, _decisions.csv
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from brux import emg  # noqa: E402
from snore import visualize  # noqa: E402,F401  (한글 글꼴 설정)
from trigger.link import MarkerOutput  # noqa: E402
from trigger.safety import SafetyConfig, SafetyGate  # noqa: E402


def read_emg(path, column=None):
    if path.endswith(".npy"):
        x = np.load(path).astype(float)
        return x[:, -1] if x.ndim == 2 else x          # [EEG, 턱EMG] 2열이면 턱
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        col = column or next(c for c in rd.fieldnames if c.lower() not in ("t", "t_s", "time", "us", "t_us"))
        return np.array([float(r[col]) for r in rd if r[col] not in ("", "nan")])


def plot(r, fs, truth, path, title):
    fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 0.6]})
    k = max(1, len(r["y"]) // 40000)
    ax[0].plot(np.arange(0, len(r["y"]), k) / fs, r["y"][::k], lw=0.3, color="#555")
    ax[0].set_ylabel("필터 후 µV")
    ax[0].set_title(title)
    n = r["norm"]
    scale = (100 / n.mvc_uv) if n.mvc_uv else 1.0
    unit = "%MVC" if n.mvc_uv else "µV"
    ax[1].plot(r["t"], r["env"] * scale, lw=0.6, color="#4c78a8", label="근육 세기")
    ax[1].plot(r["t"], r["thr"] * scale, lw=1, ls="--", color="#e4572e", label="기준")
    ax[1].plot(r["t"], r["bg"] * scale, lw=0.8, color="#999", label="배경")
    ax[1].set_ylabel(unit)
    if r["bad"].any():
        ax[1].fill_between(r["t"], 0, (r["env"] * scale).max(), where=r["bad"], color="gray", alpha=0.3, label="품질 불량(제외)")
    ax[1].legend(loc="upper right", fontsize=8)
    for s, e, _ in r["bursts"]:
        ax[2].plot([s, e], [0.3, 0.3], color="#4c78a8", lw=6, solid_capstyle="butt")
    col = {"위상성": "#e4572e", "긴장성": "#7b2d26", "혼합": "#f2a541"}
    for s, e, kind, nb, _ in r["episodes"]:
        ax[2].plot([s, e], [0.7, 0.7], color=col[kind], lw=6, solid_capstyle="butt")
        ax[2].text(s, 0.85, kind, fontsize=8)
    for s, e, *_ in truth or []:
        for a in ax[:2]:
            a.axvspan(s, e, color="#e4572e", alpha=0.1)
    ax[2].set_yticks([0.3, 0.7], ["버스트", "에피소드"])
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("시간 (초)   ·   연한 빨강 = 정답 구간")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emg")
    ap.add_argument("--calib", help="brux_calibrate.py 결과 json (MVC)")
    ap.add_argument("--column")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--warmup", type=float, default=None, help="시작 유예 (분). 기본: 데모 0, 실제 20")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    truth, mvc, fs = None, None, a.fs or emg.FS
    if a.demo:
        from brux import synth
        x, truth = synth.make(minutes=20)
        mvc = emg.mvc_from_calibration(synth.make_calibration())
        name = "demo"
    else:
        if not a.emg:
            raise SystemExit("--emg 파일 또는 --demo 가 필요해요")
        x = read_emg(a.emg, a.column)
        side = os.path.splitext(a.emg)[0] + ".json"
        if not a.fs and os.path.exists(side):
            fs = float(json.load(open(side)).get("fs", fs))
        name = os.path.splitext(os.path.basename(a.emg))[0]
        if name == "emg":                                       # 동기 세션 폴더의 emg.npy
            name = os.path.basename(os.path.dirname(os.path.abspath(a.emg)))
        if a.calib:
            mvc = json.load(open(a.calib))["mvc_uv"]

    r = emg.analyze(x, fs, mvc_uv=mvc)
    m = r["metrics"]
    print(f"기록 {len(x) / fs / 60:.1f}분 · {fs:.0f} Hz · 기준: {r['norm'].text()}")
    print(f"신호 품질: 포화·멈춤 {r['sat_ratio'] * 100:.1f}% · 전원잡음 비율 {r['hum_ratio']:.2f}")
    if r["sat_ratio"] > 0.05:
        print("※ 포화가 5% 넘음: 전극 접촉 / 기준 전극(귓불) 연결 / 게인을 확인하세요")
    if r["hum_ratio"] > 1.0:
        print("※ 전원 잡음이 근전도보다 큼: 노트북 배터리 + 아이솔레이터, 선 짧게")
    if not mvc:
        print("※ 보정(MVC)이 없어서 배경 2배 기준만 사용. brux_calibrate.py 로 보정하면 더 정확해요")
    print()

    warm = a.warmup if a.warmup is not None else (0 if a.demo else 20)
    gate = SafetyGate(SafetyConfig(armed=False, warmup_min=warm, min_conf=0.5),
                      log_path=os.path.join(a.out, f"brux_{name}_decisions.csv"), on_send=MarkerOutput())
    if os.path.exists(gate.log_path):
        os.remove(gate.log_path)
    for s, e, kind, nb, peak in r["episodes"]:
        i = (r["t"] >= s) & (r["t"] <= e)
        conf = float(np.clip(0.5 + (peak - 1) / 4, 0, 1))        # 기준의 1배 = 0.5, 3배 이상 = 1.0
        print(f"[{int(s) // 3600:02d}:{int(s) % 3600 // 60:02d}:{int(s) % 60:02d}] {kind} 에피소드  "
              f"{e - s:4.1f}초  버스트 {nb}개  최대 기준의 {peak:.1f}배")
        gate.request(e, "BRUX", conf, not r["bad"][i].any(), {"kind": kind, "bursts": nb})

    print("\n── 이갈이 측정 요약")
    for k, v in m.items():
        print(f"  {k:22s} {v}")
    with open(os.path.join(a.out, f"brux_{name}_summary.csv"), "w", encoding="utf-8-sig") as f:
        f.write("항목,값\n" + "".join(f"{k},{v}\n" for k, v in m.items()))
    with open(os.path.join(a.out, f"brux_{name}_episodes.csv"), "w", encoding="utf-8-sig") as f:
        f.write("start_s,end_s,kind,bursts,peak_x_threshold\n")
        for s, e, kind, nb, pk in r["episodes"]:
            f.write(f"{s:.2f},{e:.2f},{kind},{nb},{pk:.2f}\n")

    if truth:
        hit = sum(any(es < te and ee > ts for es, ee, *_ in r["episodes"]) for ts, te, _ in truth)
        fa = sum(not any(es < te and ee > ts for ts, te, _ in truth) for es, ee, *_ in r["episodes"])
        print(f"\n정답 에피소드 {len(truth)}개 중 {hit}개 탐지 · 잘못 탐지 {fa}개")
    s = gate.summary()
    print(f"자극 요청 판단: 마커 {s['마커']} · 보류 {s['보류']}")
    path = os.path.join(a.out, f"brux_{name}.png")
    plot(r, fs, truth, path, f"이갈이 탐지 — {name}")
    print(f"그림: {path}")


if __name__ == "__main__":
    main()
