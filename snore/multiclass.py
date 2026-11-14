"""소리 종류 분류 (코골이 / 이불 코골이 / 옷·이불 / 뒤척임 / 말소리 / 조용함 …)

실제 밤 녹음에는 여러 소리가 섞여 있음 → 0.5초마다 "지금 무슨 소리인지" 를 붙여서
  - 밤새 소리 종류별 시간 (코골이 몇 분, 뒤척임 몇 분 …)
  - 코골이를 놓쳤다면 무엇으로 착각했는지 (혼동행렬)
을 보여 준다.

코골이 확정·트리거 판단은 여전히 이진 모델(코골이 확률) + 반복 규칙이 담당하고,
이 모델은 '설명용'(무슨 소리였나)으로 함께 쓴다.
"""
import numpy as np
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier

from .config import LABEL_KO


class MultiForest:
    def __init__(self, seed=0):
        self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                          n_jobs=-1, random_state=seed)
        self.classes = []

    def fit(self, F, labels, w=None):
        self.clf.fit(F, np.asarray(labels, dtype=str), sample_weight=w)
        self.classes = list(self.clf.classes_)
        return self

    def proba(self, F):
        return self.clf.predict_proba(F)

    def predict(self, F, smooth=3):
        """창별 소리 종류. 확률을 앞뒤 창과 평균내서 흔들림 줄임"""
        if len(F) == 0:
            return np.array([], dtype=object)
        P = self.proba(F)
        if smooth > 1:
            P = median_filter(P, size=(smooth, 1), mode="nearest")
        return np.array(self.classes, dtype=object)[P.argmax(axis=1)]


def segments(t, labels, hop=0.5):
    """같은 종류가 이어진 창 → [(시작, 끝, 종류)]  (창 하나 = hop 초 몫)"""
    out = []
    for ti, lab in zip(t, labels):
        ti = float(ti)
        if out and out[-1][2] == lab and abs(ti - out[-1][1]) < 1e-6:
            out[-1] = (out[-1][0], ti + hop, lab)
        else:
            out.append((ti, ti + hop, lab))
    return out


def time_by_label(labels, hop=0.5):
    labs, cnt = np.unique(labels, return_counts=True)
    return {LABEL_KO.get(l, l): c * hop for l, c in sorted(zip(labs, cnt), key=lambda x: -x[1])}
