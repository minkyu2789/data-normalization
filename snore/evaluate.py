"""평가

- 참가자 단위 교차검증 (Leave-One-Subject-Out)
  한 사람을 빼고 나머지로 학습 → 뺀 사람으로 시험. 이걸 사람 수만큼 반복.
  '처음 보는 사람에게도 통하는가'를 확인하는 방법 (같은 사람 데이터가 학습·시험에 섞이면 점수가 부풀려짐)

- 개인 적응 (personalization)
  시험 대상의 앞 30% 데이터를 '보정용'으로 학습에 추가 (가중치 3배) → 나머지 70%로 시험.
  (시간 때문에 랜덤포레스트만)
  일반 모델과 같은 70%에서 비교해서 '개인 데이터를 조금 넣으면 얼마나 좋아지는가'를 봄.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .config import PROB_THR
from .models import make_model
from .postprocess import smooth


def scores(y, p, thr=PROB_THR):
    pred = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "cm": confusion_matrix(y, pred, labels=[0, 1]),
    }


def smooth_by_clip(ws, p):
    out = p.copy()
    for c in np.unique(ws.clip):
        i = np.where(ws.clip == c)[0]
        i = i[np.argsort(ws.t[i])]
        out[i] = smooth(p[i])
    return out


def per_label(ws, p, thr=PROB_THR):
    """원래 소리 종류별로 '코골이'라고 판단한 비율"""
    return {lab: float(np.mean(p[ws.label == lab] >= thr)) for lab in np.unique(ws.label)}


def _fit(kind, tr, w=None, seed=0, epochs=None):
    m = make_model(kind, seed)
    if epochs and kind == "cnn":
        m.epochs = epochs
    return m.fit(tr.F, tr.M.astype(np.float32) if len(tr.M) else tr.M, tr.y, w)


def _pred(m, te):
    return m.predict_proba(te.F, te.M.astype(np.float32) if len(te.M) else te.M)


def loso(ws, kinds, personalize=True, calib_frac=0.3, log=print, cnn_epochs=None):
    subs = sorted(np.unique(ws.subject))
    rows, all_pred = [], {k: np.full(len(ws.y), np.nan) for k in kinds}
    for s in subs:
        tr = ws.take(ws.subject != s)
        te_idx = np.where((ws.subject == s) & (ws.aug == 0))[0]
        te = ws.take(te_idx)
        for k in kinds:
            m = _fit(k, tr, epochs=cnn_epochs)
            p = smooth_by_clip(te, _pred(m, te))
            all_pred[k][te_idx] = p
            r = scores(te.y, p)
            rows.append({"subject": s, "model": k, "mode": "일반", **{x: r[x] for x in r if x != "cm"}})
            log(f"  {s} · {k:4s} · 일반      F1 {r['f1']:.3f}  정확도 {r['accuracy']:.3f}")

            if personalize and k == "rf":
                # 각 클립의 앞 30%를 보정용으로
                cal = np.zeros(len(ws.y), bool)
                clips = np.unique(ws.clip[ws.subject == s])
                short = np.mean([len(np.unique(ws.t[ws.clip == c])) <= 2 for c in clips]) > 0.5
                if short:
                    # 1초짜리 클립(공개 데이터 등): 클립 단위로 30%를 보정용으로 (종류별로 고르게)
                    rng = np.random.default_rng(0)
                    for lab in np.unique(ws.label[ws.subject == s]):
                        cl = np.unique(ws.clip[(ws.subject == s) & (ws.label == lab)])
                        pick = rng.choice(cl, size=max(1, int(len(cl) * calib_frac)), replace=False)
                        cal[np.isin(ws.clip, pick)] = True
                else:
                    for c in clips:
                        ci = np.where(ws.clip == c)[0]
                        tmax = ws.t[ci].max()
                        cal[ci[ws.t[ci] <= ws.t[ci].min() + calib_frac * (tmax - ws.t[ci].min())]] = True
                rest_idx = np.where((ws.subject == s) & (ws.aug == 0) & ~cal)[0]
                if len(rest_idx) == 0:
                    continue
                rest = ws.take(rest_idx)
                tr2_idx = np.where((ws.subject != s) | cal)[0]
                tr2 = ws.take(tr2_idx)
                w = np.where(tr2.subject == s, 3.0, 1.0)
                m2 = _fit(k, tr2, w, epochs=cnn_epochs)
                p_gen = smooth_by_clip(rest, _pred(m, rest))
                p_per = smooth_by_clip(rest, _pred(m2, rest))
                rg, rp = scores(rest.y, p_gen), scores(rest.y, p_per)
                rows.append({"subject": s, "model": k, "mode": "일반(70%)", **{x: rg[x] for x in rg if x != "cm"}})
                rows.append({"subject": s, "model": k, "mode": "개인적응(70%)", **{x: rp[x] for x in rp if x != "cm"}})
                log(f"  {s} · {k:4s} · 개인적응  F1 {rg['f1']:.3f} → {rp['f1']:.3f}")
    return rows, all_pred
