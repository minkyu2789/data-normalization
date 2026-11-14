"""코골이 탐지 학습·평가

    python train.py --demo                  예시 데이터로 전체 과정 실행
    python train.py --data data             실제 녹음으로 실행
    python train.py --models rule,rf        CNN 없이 (빠름)

결과: results/ 폴더 (표, 그림, 학습된 모델, 자동 리포트)
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from snore import visualize as viz
from snore.config import LABEL_KO
from snore.dataio import load_dataset, summary_table
from snore.evaluate import loso, per_label, scores
from snore.models import HAS_TORCH, make_model
from snore.multiclass import MultiForest
from snore.pipeline import build

KO = {"rule": "규칙 기반", "rf": "랜덤포레스트", "cnn": "CNN"}


def line(title=""):
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true", help="data 폴더에 예시 데이터를 만들고 실행")
    ap.add_argument("--models", default="rule,rf,cnn")
    ap.add_argument("--aug", type=int, default=2, help="녹음 하나당 증강본 개수")
    ap.add_argument("--cnn-epochs", type=int, default=15)
    ap.add_argument("--no-personal", action="store_true", help="개인 적응 실험 생략")
    a = ap.parse_args()

    kinds = [k.strip() for k in a.models.split(",") if k.strip()]
    if "cnn" in kinds and not HAS_TORCH:
        print("※ torch가 없어서 CNN은 건너뜁니다 (pip install torch)")
        kinds.remove("cnn")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    if a.demo and not os.path.isdir(os.path.join(a.data, "subjects")):
        from snore.synth import make_demo
        print("예시 데이터 생성 중…")
        make_demo(a.data)

    # 1. 데이터
    line("1. 데이터셋")
    clips = load_dataset(a.data)
    if not clips:
        raise SystemExit(f"{a.data} 에 녹음이 없어요. README의 폴더 구조를 확인하세요.")
    subs, labs, tab = summary_table(clips)
    df_sum = pd.DataFrame(tab).T.round(1)
    df_sum.columns = [LABEL_KO.get(c, c) for c in df_sum.columns]
    df_sum["합계(초)"] = df_sum.sum(axis=1)
    print(df_sum.to_string())
    df_sum.to_csv(os.path.join(a.out, "dataset_summary.csv"), encoding="utf-8-sig")

    # 2. 전처리 + 특징
    line("2. 전처리 · 특징 추출")
    ws = build(clips, n_aug=a.aug, with_mel="cnn" in kinds)
    n0 = int((ws.aug == 0).sum())
    print(f"1초 창 {n0}개 (원본)  +  증강 {len(ws.y) - n0}개   ·   코골이 비율 {ws.y[ws.aug == 0].mean() * 100:.0f}%")
    print(f"특징 {ws.F.shape[1]}개" + (f"  ·  멜 스펙트로그램 {ws.M.shape[1]}×{ws.M.shape[2]}" if len(ws.M) else ""))

    # 3. 평가
    line("3. 참가자 단위 교차검증")
    rows, preds = loso(ws, kinds, personalize=not a.no_personal, cnn_epochs=a.cnn_epochs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "cv_results.csv"), index=False, encoding="utf-8-sig")

    line("결과 요약 (처음 보는 사람 기준, 평균)")
    g = df[df["mode"] == "일반"].groupby("model")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
    summary = pd.DataFrame({
        "모델": [KO[k] for k in g.index],
        "정확도": [f"{m:.3f} ± {s:.3f}" for m, s in g["accuracy"].values],
        "정밀도": [f"{m:.3f}" for m in g["precision"]["mean"]],
        "재현율": [f"{m:.3f}" for m in g["recall"]["mean"]],
        "F1": [f"{m:.3f} ± {s:.3f}" for m, s in g["f1"].values],
    })
    print(summary.to_string(index=False))
    pers = None
    if not a.no_personal and (df["mode"] == "개인적응(70%)").any():
        p = df[df["mode"].isin(["일반(70%)", "개인적응(70%)"])].groupby(["model", "mode"])["f1"].mean().unstack()
        pers = pd.DataFrame({"모델": [KO[k] for k in p.index], "일반 F1": p["일반(70%)"].round(3).values,
                             "개인적응 F1": p["개인적응(70%)"].round(3).values})
        print("\n개인 적응 (시험 대상의 앞 30%를 학습에 추가)")
        print(pers.to_string(index=False))

    orig = ws.aug == 0
    wo = ws.take(np.where(orig)[0])
    cms, rates = {}, {}
    for k in kinds:
        p = preds[k][orig]
        cms[k] = scores(wo.y, p)["cm"]
        rates[k] = per_label(wo, p)
    line("소리 종류별 '코골이'로 판단한 비율")
    rt = pd.DataFrame(rates).rename(columns=KO, index=lambda x: LABEL_KO.get(x, x)).round(2)
    print(rt.to_string())

    # 3-2. 소리 종류 분류 (섞인 밤 녹음에서 '무슨 소리였나')
    line("3-2. 소리 종류 분류 (랜덤포레스트, 참가자 단위 교차검증)")
    lab_pred = np.full(len(ws.y), "", dtype=object)
    for s in sorted(np.unique(ws.subject)):
        tr = ws.take(np.where(ws.subject != s)[0])
        te_idx = np.where((ws.subject == s) & orig)[0]
        if len(np.unique(tr.label)) < 2 or not len(te_idx):
            continue
        mf = MultiForest().fit(tr.F, tr.label)
        lab_pred[te_idx] = mf.predict(ws.F[te_idx], smooth=1)
    ok = orig & (lab_pred != "")
    order = [l for l in LABEL_KO if l in set(ws.label[ok])] + sorted(set(ws.label[ok]) - set(LABEL_KO))
    from sklearn.metrics import confusion_matrix as _cm
    cm6 = _cm(ws.label[ok].astype(str), lab_pred[ok].astype(str), labels=order)
    acc6 = float(np.trace(cm6) / max(cm6.sum(), 1))
    per = {LABEL_KO.get(l, l): round(cm6[i, i] / max(cm6[i].sum(), 1), 3) for i, l in enumerate(order)}
    print(f"전체 정확도 {acc6:.3f}")
    for k, v in per.items():
        print(f"  {k:12s} 맞힌 비율 {v:.3f}")

    # 4. 그림
    line("4. 그림 저장")
    viz.confusion_multi(cm6, order, os.path.join(a.out, "fig7_sound_types.png"))
    viz.class_examples(clips, os.path.join(a.out, "fig1_patterns.png"))
    viz.feature_box(ws, os.path.join(a.out, "fig2_features.png"))
    viz.confusion(cms, os.path.join(a.out, "fig3_confusion.png"))
    viz.model_bars(rows, os.path.join(a.out, "fig4_models.png"))
    viz.label_rates(rates, os.path.join(a.out, "fig5_by_sound.png"))

    # 5. 전체 데이터로 최종 모델
    line("5. 최종 모델 학습 (전체 데이터)")
    saved = {}
    for k in kinds:
        m = make_model(k)
        if k == "cnn":
            m.epochs = a.cnn_epochs
        m.fit(ws.F, ws.M.astype(np.float32) if len(ws.M) else ws.M, ws.y)
        if k == "rule":
            with open(os.path.join(a.out, "model_rule.json"), "w") as f:
                json.dump({"db": m.db, "low": m.low, "flat": m.flat}, f)
            print("규칙:", m.describe())
        elif k == "rf":
            joblib.dump(m.clf, os.path.join(a.out, "model_rf.joblib"))
            viz.importance(m.importance(), os.path.join(a.out, "fig6_importance.png"))
            print("중요 특징:", ", ".join(n for n, _ in m.importance()[:5]))
        elif k == "cnn":
            import torch
            torch.save(m.net.state_dict(), os.path.join(a.out, "model_cnn.pt"))
        saved[k] = True
    mf = MultiForest().fit(ws.F, ws.label)
    joblib.dump(mf, os.path.join(a.out, "model_rf_multi.joblib"))
    print("저장:", ", ".join(f"model_{k}" for k in saved), "+ model_rf_multi (소리 종류)")

    # 6. 자동 리포트
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as f:
        f.write("# 코골이 탐지 결과\n\n## 데이터셋 (초)\n\n" + df_sum.to_markdown() + "\n\n")
        f.write(f"- 1초 창 {n0}개, 증강 포함 {len(ws.y)}개, 코골이 비율 {ws.y[orig].mean() * 100:.0f}%\n\n")
        f.write("## 모델 비교 (참가자 단위 교차검증)\n\n" + summary.to_markdown(index=False) + "\n\n")
        if pers is not None:
            f.write("## 개인 적응\n\n" + pers.to_markdown(index=False) + "\n\n")
        f.write("## 소리 종류별 '코골이' 판단 비율\n\n" + rt.to_markdown() + "\n\n")
        f.write(f"## 소리 종류 분류 (6종)\n\n전체 정확도 {acc6:.3f}\n\n" +
                pd.Series(per, name="맞힌 비율").to_frame().to_markdown() + "\n\n![](fig7_sound_types.png)\n\n")
        f.write("![](fig1_patterns.png)\n![](fig2_features.png)\n![](fig3_confusion.png)\n"
                "![](fig4_models.png)\n![](fig5_by_sound.png)\n")
    print(f"\n완료 · {time.time() - t0:.0f}초 · 결과 폴더: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
