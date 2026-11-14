"""긴 녹음에서 코골이 탐지 → 트리거까지 (실시간처럼 0.5초씩 처리하며 콘솔 출력)

    python detect.py --wav data/test/P99_night.wav --labels data/test/P99_night.txt
    python detect.py --wav 내녹음.wav --model cnn
"""
import argparse
import json
import os

import joblib
import numpy as np

from snore import visualize as viz
from snore.config import PROB_THR, is_snore
from snore.dataio import load_wav, read_audacity_labels
from snore.evaluate import scores
from snore.models import I_PEAK, CNNModel, ForestModel, RuleModel
from snore.pipeline import SOUND_MIN_DB, windows_from_audio
from snore.postprocess import StreamDetector, TriggerController, episodes, smooth, to_events
from snore.measure import print_metrics, snore_metrics
from snore.multiclass import segments, time_by_label
from snore.config import LABEL_KO


def load_model(kind, d):
    if kind == "rule":
        m = RuleModel()
        m.__dict__.update(json.load(open(os.path.join(d, "model_rule.json"))))
    elif kind == "rf":
        m = ForestModel()
        m.clf = joblib.load(os.path.join(d, "model_rf.joblib"))
    else:
        import torch
        m = CNNModel()
        m.net.load_state_dict(torch.load(os.path.join(d, "model_cnn.pt")))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--labels", help="Audacity 라벨 (있으면 정확도도 계산)")
    ap.add_argument("--model", default="rf", choices=["rule", "rf", "cnn"])
    ap.add_argument("--models-dir", default="results")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    x = load_wav(a.wav)
    m = load_model(a.model, a.models_dir)
    F, M, t, floor = windows_from_audio(x, with_mel=a.model == "cnn")
    prob = m.predict_proba(F, M.astype(np.float32) if len(M) else M)
    print(f"녹음 {len(x) / 16000:.0f}초 · 창 {len(t)}개 · 바닥소음 {floor:.1f} dBFS · 모델 {a.model}\n")

    def hardware_trigger(tt, info):
        # 하드웨어 팀 연결 지점: 여기서 ESP32로 트리거 명령을 보냄 (지금은 기록만)
        pass

    ctl = TriggerController(on_trigger=hardware_trigger)
    det = StreamDetector(ctl)
    for ti, pi in zip(t, prob):
        det.push(float(ti), float(pi))

    sm = smooth(prob)
    ev = to_events(t, sm >= PROB_THR, prob)
    eps = episodes(ev)
    trig = [l for l in ctl.log if l["action"] == "트리거"]
    print(f"\n요약: 코골이 후보 소리 {len(ev)}개 · 코골이 확정 구간 {len(eps)}개 · "
          f"트리거 {len(trig)}회 · 보류 {len(ctl.log) - len(trig)}회")

    # 측정 요약
    m = snore_metrics(t, sm >= PROB_THR, ev, eps, F, floor, len(x) / 16000)
    print()
    print_metrics(m)

    # 소리 종류 (섞인 녹음에서 무슨 소리였나)
    kinds = None
    mp = os.path.join(a.models_dir, "model_rf_multi.joblib")
    if os.path.exists(mp):
        mf = joblib.load(mp)
        kinds = mf.predict(F)
        # 방 소음보다 6 dB 이상 큰 소리가 없고 코골이 확률도 낮으면 조용함 (작은 이불 코골이는 남김)
        kinds[(F[:, I_PEAK] < SOUND_MIN_DB) & (sm < PROB_THR)] = "quiet"
        print("\n── 소리 종류별 시간 (초)")
        for k, v in time_by_label(kinds).items():
            print(f"  {k:14s} {v:7.1f}")

    truth = read_audacity_labels(a.labels) if a.labels else None
    if truth:
        # 정답도 학습 때와 같은 규칙: 코골이 구간 안에서 소리가 있는 창만 '코골이', 숨 사이 조용한 창은 평가에서 제외
        y = np.zeros(len(t), int)
        use = np.ones(len(t), bool)
        loud = F[:, I_PEAK] >= SOUND_MIN_DB
        for s, e, lab in truth:
            if is_snore(lab):
                inside = (t + 0.5 >= s) & (t + 0.5 < e)
                y[inside & loud] = 1
                use[inside & ~loud] = False
        r = scores(y[use], sm[use])
        print(f"창 단위: 정확도 {r['accuracy']:.3f} · 정밀도 {r['precision']:.3f} · "
              f"재현율 {r['recall']:.3f} · F1 {r['f1']:.3f}")
        gt = [(s, e) for s, e, lab in truth if is_snore(lab)]
        hit = sum(any(es < e and ee > s for es, ee, _ in eps) for s, e in gt)
        fa = sum(not any(es < e and ee > s for s, e in gt) for es, ee, _ in eps)
        print(f"구간 단위: 실제 코골이 구간 {len(gt)}개 중 {hit}개 탐지 · 잘못 확정한 구간 {fa}개")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"timeline_{os.path.splitext(os.path.basename(a.wav))[0]}_{a.model}.png")
    viz.timeline(x, t, prob, sm, ev, eps, ctl.log, truth, path, kinds=kinds)
    base = path[:-4]
    with open(base + "_summary.csv", "w", encoding="utf-8-sig") as f:
        f.write("항목,값\n" + "".join(f"{k},{v}\n" for k, v in m.items()))
    if kinds is not None:
        with open(base + "_sounds.csv", "w", encoding="utf-8-sig") as f:
            f.write("start_s,end_s,label,label_ko\n")
            for s0, e0, lab in segments(t, kinds):
                f.write(f"{s0:.1f},{e0:.1f},{lab},{LABEL_KO.get(lab, lab)}\n")
    with open(path.replace(".png", "_triggers.csv"), "w", encoding="utf-8-sig") as f:
        f.write("t_s,action,reason,repeats,prob\n")
        for l in ctl.log:
            f.write(f"{l['t']:.1f},{l['action']},{l['reason']},{l.get('repeats', '')},{l.get('prob', '')}\n")
    print(f"그림: {path}")


if __name__ == "__main__":
    main()
