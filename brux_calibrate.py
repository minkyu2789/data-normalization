"""이갈이 보정 (사람마다 1번, 측정 전에 깨어 있는 상태로 약 3분)

1) 안내 모드: 측정 프로그램(collect.py)으로 기록을 시작한 '직후' 이것을 실행해서 화면 안내대로 따라 하기
       python brux_calibrate.py --guide --subject P01
   → 각 동작의 시작·끝 시각이 calib_P01_marks.csv 로 저장됨 (기록 시작 기준 초)
     기록 시작과 안내 시작 사이 몇 초 차이는 --offset 으로 보정

2) 분석 모드: 보정 기록 + 표시 파일 → 기준값 계산 + 탐지기가 제대로 구분하는지 확인
       python brux_calibrate.py --emg sessions/…/emg.npy --marks calib_P01_marks.csv --subject P01
   → results/brux_calib_P01.json (MVC, 배경, 기준) — brux_detect.py --calib 로 사용
   확인 항목: 휴식에서 오탐 0, 꽉 물기 = 긴장성, 이 갈기 = 위상성, 삼키기·말하기 = 에피소드 아님

동작 순서 (안내 모드가 같은 순서로 알려 줌)
  rest       휴식, 턱 힘 빼기             60초
  clench1~3  최대로 꽉 물기 (사이 5초 쉼)  3초씩
  grind      이 가는 흉내 (좌우로)         10초
  tap        이를 딱딱 부딪히기            10초
  hold       살짝 계속 물고 있기           10초
  swallow    침 삼키기 5번 (4초 간격)       20초
  talk       말하기 (숫자 세기)            10초
  yawn       하품 / 입 크게 벌리기         10초
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from brux import emg

PROTOCOL = [("rest", 60, "턱에 힘을 빼고 가만히 있으세요"),
            ("pause", 5, "쉬기"), ("clench1", 3, "최대로 꽉 무세요!"), ("pause", 5, "쉬기"),
            ("clench2", 3, "최대로 꽉 무세요!"), ("pause", 5, "쉬기"),
            ("clench3", 3, "최대로 꽉 무세요!"), ("pause", 10, "쉬기"),
            ("grind", 10, "이를 좌우로 가는 흉내"), ("pause", 10, "쉬기"),
            ("tap", 10, "이를 딱딱 부딪히기"), ("pause", 10, "쉬기"),
            ("hold", 10, "살짝 계속 물고 있기 (중간 힘)"), ("pause", 10, "쉬기"),
            ("swallow", 20, "침 삼키기 5번 (천천히, 4초 간격)"), ("pause", 5, "쉬기"),
            ("talk", 10, "숫자 1부터 소리 내어 세기"), ("pause", 5, "쉬기"),
            ("yawn", 10, "하품하거나 입을 크게 벌리기"), ("pause", 5, "끝")]
EXPECT = {"grind": "위상성", "tap": "위상성", "hold": "긴장성", "clench": "긴장성"}
NOT_BRUX = ("rest", "swallow", "talk", "yawn")


def guide(subject, out):
    path = os.path.join(out, f"calib_{subject}_marks.csv")
    os.makedirs(out, exist_ok=True)
    rows, t0 = [], time.time()
    print("기록을 시작한 상태에서 Enter 를 누르면 안내를 시작합니다.")
    input()
    t0 = time.time()
    for name, dur, msg in PROTOCOL:
        s = time.time() - t0
        print(f"\n[{s:5.1f}s] ▶ {msg}  ({dur}초)")
        for k in range(dur, 0, -1):
            print(f"   {k:2d}", end="\r", flush=True)
            time.sleep(1)
        if name != "pause":
            rows.append((round(s, 2), round(time.time() - t0, 2), name))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label"])
        w.writerows(rows)
    print(f"\n완료. 표시 파일: {path}  → 기록을 멈추고 분석 모드로 실행하세요")


def read_marks(path, offset=0.0):
    with open(path, encoding="utf-8-sig") as f:
        return [(float(r["start_s"]) + offset, float(r["end_s"]) + offset, r["label"]) for r in csv.DictReader(f)]


def analyze_calib(x, fs, marks, subject, out):
    clench = [(s, e) for s, e, l in marks if l.startswith("clench")]
    rest = [(s, e) for s, e, l in marks if l == "rest"]
    mvc = emg.mvc_from_calibration(x, fs, clench or None)
    r = emg.analyze(x, fs, mvc_uv=mvc)
    t, env = r["t"], r["env"]
    rest_uv = float(np.median(np.concatenate([env[(t >= s) & (t <= e)] for s, e in rest]))) if rest else float(np.percentile(env, 30))
    print(f"MVC {mvc:.1f} µV · 휴식 {rest_uv:.1f} µV · 비율 {mvc / max(rest_uv, 1e-6):.0f}배 · 포화 {r['sat_ratio'] * 100:.1f}% · 전원잡음 비율 {r['hum_ratio']:.2f}")
    warn = []
    if mvc / max(rest_uv, 1e-6) < 5:
        warn.append("꽉 물기와 휴식 차이가 5배 미만 → 전극 위치·접촉 확인")
    if r["sat_ratio"] > 0.01:
        warn.append("포화 있음 → 기준 전극(귓불) 연결·게인 확인")
    if r["hum_ratio"] > 1.0:
        warn.append("전원 잡음이 근전도보다 큼 → 노트북 배터리, 아이솔레이터, 선 짧게")

    print("\n동작별 확인")
    ok_all = True
    for s, e, lab in marks:
        found = [ep for ep in r["episodes"] if ep[0] < e + 1 and ep[1] > s - 1]
        base = lab.rstrip("0123456789")
        if base in NOT_BRUX:
            ok = not found
            want = "에피소드 없어야 함"
        elif base.startswith("clench"):
            ok = bool(r["bursts"]) and any(b[0] < e and b[1] > s for b in r["bursts"])
            want = "버스트 있어야 함 (3초짜리라 에피소드는 긴장성)"
        else:
            ok = any(ep[2] in (EXPECT.get(base), "혼합") for ep in found)
            want = f"{EXPECT.get(base)} 에피소드"
        ok_all &= ok
        got = ", ".join(f"{ep[2]}({ep[1] - ep[0]:.1f}초)" for ep in found) or "없음"
        print(f"  {'✓' if ok else '✕'} {lab:9s} {s:6.1f}–{e:6.1f}s  기대: {want:28s} 결과: {got}")
    res = {"subject": subject, "fs": fs, "mvc_uv": round(mvc, 2), "rest_uv": round(rest_uv, 2),
           "mvc_thr": emg.MVC_THR, "bg_mult": emg.BG_MULT, "all_ok": bool(ok_all), "warnings": warn}
    path = os.path.join(out, f"brux_calib_{subject}.json")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(path, "w"), ensure_ascii=False, indent=2)
    for w in warn:
        print("※", w)
    print(f"\n{'보정 통과' if ok_all else '일부 동작이 기대와 다름 (위 ✕ 확인)'} → {path}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--emg")
    ap.add_argument("--marks")
    ap.add_argument("--offset", type=float, default=0.0, help="표시 시각에 더할 초 (기록 시작과 안내 시작 차이)")
    ap.add_argument("--fs", type=int, default=emg.FS)
    ap.add_argument("--subject", default="P01")
    ap.add_argument("--demo", action="store_true", help="가짜 보정 기록으로 시험")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.guide:
        return guide(a.subject, a.out)
    if a.demo:
        from brux.synth import make_protocol
        x, marks = make_protocol()
        return analyze_calib(x, emg.FS, marks, "demo", a.out)
    if not (a.emg and a.marks):
        raise SystemExit("--guide 또는 --emg 와 --marks 가 필요해요 (--demo 로 시험 가능)")
    from brux_detect import read_emg
    analyze_calib(read_emg(a.emg), a.fs, read_marks(a.marks, a.offset), a.subject, a.out)


if __name__ == "__main__":
    main()
