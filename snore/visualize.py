"""그림 저장"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .config import LABEL_KO, SR  # noqa: E402
from .features import FEATURE_NAMES, logmel  # noqa: E402
from .preprocess import bandpass  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

C_SNORE, C_NOISE = "#e4572e", "#4c78a8"
MODEL_KO = {"rule": "규칙 기반", "rf": "랜덤포레스트", "cnn": "CNN"}


def ko(lab):
    return LABEL_KO.get(lab, lab)


def class_examples(clips, path, sec=6):
    """소리 종류별 파형 + 스펙트로그램 (패턴 비교). 짧은 클립은 그 길이만큼만 그림"""
    by = {}
    for c in clips:
        by.setdefault(c.label, c)
    labs = list(by)
    fig, ax = plt.subplots(2, len(labs), figsize=(3.2 * len(labs), 5.2), squeeze=False,
                           gridspec_kw={"height_ratios": [1, 2]})
    xs = {lab: bandpass(by[lab].audio[: int(sec * SR)]) for lab in labs}
    ymax = max(np.abs(v).max() for v in xs.values()) * 1.05
    for j, lab in enumerate(labs):
        x = xs[lab]
        dur = len(x) / SR
        t = np.arange(len(x)) / SR
        ax[0, j].plot(t, x, lw=0.4, color=C_SNORE if "snore" in lab or "코골이" in lab else C_NOISE)
        ax[0, j].set_title(ko(lab), fontsize=11)
        ax[0, j].set_xlim(0, dur)
        ax[0, j].set_ylim(-ymax, ymax)          # 같은 크기 기준으로 비교
        ax[0, j].set_yticks([])
        m = logmel(x[None, :])[0]
        ax[1, j].imshow(m, origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, 64])
        ax[1, j].set_xlabel("초")
        ax[1, j].set_yticks([0, 32, 64], ["50Hz", "~800Hz", "4kHz"] if j == 0 else [])
    fig.suptitle("소리 종류별 파형(위)과 멜 스펙트로그램(아래)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_box(ws, path):
    keys = ["rel_db", "low_ratio", "high_ratio", "centroid", "flatness", "zcr"]
    names = {"rel_db": "상대 크기 (dB)", "low_ratio": "저음 비율 (50–300Hz)", "high_ratio": "고음 비율 (1–4kHz)",
             "centroid": "스펙트럼 중심 (kHz)", "flatness": "평탄도 (잡음형↑)", "zcr": "영교차율"}
    labs = list(dict.fromkeys(ws.label[ws.aug == 0]))
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, k in zip(ax.flat, keys):
        i = FEATURE_NAMES.index(k)
        data = [ws.F[(ws.label == l) & (ws.aug == 0), i] for l in labs]
        bp = a.boxplot(data, patch_artist=True, showfliers=False)
        for patch, l in zip(bp["boxes"], labs):
            patch.set_facecolor(C_SNORE if ("snore" in l or "코골이" in l) else C_NOISE)
            patch.set_alpha(0.7)
        a.set_xticks(range(1, len(labs) + 1), [ko(l) for l in labs], rotation=30, fontsize=9)
        a.set_title(names[k])
    fig.suptitle("특징별 분포 (빨강 = 코골이, 파랑 = 잡음)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(cms, path):
    fig, ax = plt.subplots(1, len(cms), figsize=(3.6 * len(cms), 3.4), squeeze=False)
    for a, (name, cm) in zip(ax[0], cms.items()):
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(1)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}\n({cmn[i, j] * 100:.0f}%)", ha="center", va="center",
                       color="white" if cmn[i, j] > 0.5 else "black", fontsize=10)
        a.set_xticks([0, 1], ["잡음", "코골이"])
        a.set_yticks([0, 1], ["잡음", "코골이"])
        a.set_xlabel("예측")
        a.set_ylabel("정답")
        a.set_title(MODEL_KO.get(name, name))
    fig.suptitle("혼동행렬 (처음 보는 사람으로 시험)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def model_bars(rows, path):
    import pandas as pd
    df = pd.DataFrame(rows)
    g = df[df["mode"] == "일반"].groupby("model")[["accuracy", "precision", "recall", "f1"]].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(g.columns))
    w = 0.8 / len(g)
    for k, (name, r) in enumerate(g.iterrows()):
        b = ax.bar(x + k * w - 0.4 + w / 2, r.values, w, label=MODEL_KO.get(name, name))
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, ["정확도", "정밀도", "재현율", "F1"])
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    ax.set_title("모델 비교 (참가자 단위 교차검증 평균)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def label_rates(rates, path):
    """모델별: 각 소리 종류를 '코골이'로 판단한 비율"""
    names = list(rates)
    labs = list(next(iter(rates.values())))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(labs))
    w = 0.8 / len(names)
    for k, n in enumerate(names):
        ax.bar(x + k * w - 0.4 + w / 2, [rates[n][l] for l in labs], w, label=MODEL_KO.get(n, n))
    ax.set_xticks(x, [ko(l) for l in labs])
    ax.set_ylabel("'코골이'로 판단한 비율")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    ax.legend()
    ax.set_title("소리 종류별 판단 결과 (코골이는 높을수록, 잡음은 낮을수록 좋음)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def importance(imp, path, top=12):
    imp = imp[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh([n for n, _ in imp], [v for _, v in imp], color=C_NOISE)
    ax.set_title("랜덤포레스트가 중요하게 본 특징 (상위)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


KIND_COLOR = {"snore": "#e4572e", "snore_covered": "#b8336a", "cloth": "#4c78a8", "movement": "#72b7b2",
              "speech": "#f2a541", "quiet": "#d9d9d9", "noise": "#9d755d"}


def timeline(x, t, prob, sm, events, eps, triggers, truth, path, thr=0.5, kinds=None):
    n = 4 if kinds is not None else 3
    ratios = [1.2, 1.2, 0.8] + ([0.5] if kinds is not None else [])
    fig, ax = plt.subplots(n, 1, figsize=(13, 6.5 + (1.3 if kinds is not None else 0)), sharex=True,
                           gridspec_kw={"height_ratios": ratios})
    k = max(1, len(x) // 20000)
    tt = np.arange(0, len(x), k) / SR
    ax[0].plot(tt, x[::k], lw=0.3, color="#555")
    ax[0].set_ylabel("소리")
    ax[0].set_title("탐지 타임라인")
    if truth:
        for s, e, lab in truth:
            if "snore" in lab or "코골이" in lab:
                ax[0].axvspan(s, e, color=C_SNORE, alpha=0.12)
    ax[1].plot(t, prob, lw=0.6, color="#bbb", label="원래 확률")
    ax[1].plot(t, sm, lw=1.2, color=C_SNORE, label="평활 확률")
    ax[1].axhline(thr, color="gray", ls=":", lw=0.8)
    ax[1].set_ylabel("코골이 확률")
    ax[1].legend(loc="upper right", fontsize=8)
    for s, e, *_ in events:
        ax[2].plot([s, e], [0.3, 0.3], color=C_SNORE, lw=6, solid_capstyle="butt")
    for s, e, n_ in eps:
        ax[2].plot([s, e], [0.7, 0.7], color="#7b2d26", lw=6, solid_capstyle="butt")
    for tr in triggers:
        ax[2].axvline(tr["t"], color="#2a9d8f" if tr["action"] in ("트리거", "마커", "요청") else "#bbb", lw=1.5)
    ax[2].set_yticks([0.3, 0.7], ["코골이 소리", "코골이 확정"])
    ax[2].set_ylim(0, 1)
    if kinds is not None:
        from .multiclass import segments
        seen = set()
        for s, e, lab in segments(t, kinds):
            c = KIND_COLOR.get(lab, "#888")
            ax[3].axvspan(s, e, color=c, label=None if lab in seen else ko(lab))
            seen.add(lab)
        ax[3].set_yticks([])
        ax[3].set_ylabel("소리 종류")
    ax[-1].set_xlabel("시간 (초)   ·   세로선: 초록 = 트리거, 회색 = 보류   ·   위 연한 빨강 = 정답 코골이 구간")
    if kinds is not None:
        h, l = ax[3].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=7, fontsize=9, frameon=False)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion_multi(cm, labels, path):
    """소리 종류 혼동행렬: 행 = 실제, 열 = 예측 (행마다 비율)"""
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.0 * n + 1.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{cmn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.5 else "black")
    names = [ko(l) for l in labels]
    ax.set_xticks(range(n), names, rotation=35, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("예측")
    ax.set_ylabel("실제")
    ax.set_title("소리 종류 분류 (처음 보는 사람으로 시험)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
