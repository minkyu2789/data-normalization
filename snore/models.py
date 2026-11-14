"""탐지 모델 3종 (같은 입력 → 코골이 확률 0~1)

1. RuleModel   : 규칙 기반. "충분히 크고 + 저음 비율이 높고 + 잡음형이 아니면 코골이"
                 임계값 3개를 학습 데이터에서 F1이 가장 높게 나오도록 찾음
2. ForestModel : 랜덤포레스트 (머신러닝). 손으로 만든 특징 49개 사용
3. CNNModel    : 작은 합성곱 신경망 (딥러닝). 로그 멜 스펙트로그램 + 상대 크기 사용
"""
import itertools

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .features import FEATURE_NAMES

I_REL = FEATURE_NAMES.index("rel_db")
I_LOW = FEATURE_NAMES.index("low_ratio")
I_MID = FEATURE_NAMES.index("mid_ratio")
I_FLAT = FEATURE_NAMES.index("flatness")
I_PEAK = FEATURE_NAMES.index("peak_db")


class RuleModel:
    name = "규칙 기반"

    def __init__(self):
        self.db, self.low, self.flat = 8.0, 0.6, 0.3

    def _score(self, F, db, low, flat):
        lowmid = F[:, I_LOW] + F[:, I_MID]
        return (F[:, I_REL] > db) & (lowmid > low) & (F[:, I_FLAT] < flat)

    def fit(self, F, M, y, w=None):
        best = -1
        for db, low, flat in itertools.product([4, 6, 8, 10, 12, 15], [0.5, 0.6, 0.7, 0.8, 0.9],
                                               [0.1, 0.2, 0.3, 0.4, 0.6]):
            f1 = f1_score(y, self._score(F, db, low, flat), zero_division=0)
            if f1 > best:
                best, (self.db, self.low, self.flat) = f1, (db, low, flat)
        return self

    def predict_proba(self, F, M=None):
        return self._score(F, self.db, self.low, self.flat).astype(float)

    def describe(self):
        return (f"상대 크기 > {self.db} dB  그리고  저·중음 비율 > {self.low:.1f}  "
                f"그리고  스펙트럼 평탄도 < {self.flat:.1f}")


class ForestModel:
    name = "랜덤포레스트"

    def __init__(self, seed=0):
        self.clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                          n_jobs=-1, random_state=seed)

    def fit(self, F, M, y, w=None):
        self.clf.fit(F, y, sample_weight=w)
        return self

    def predict_proba(self, F, M=None):
        return self.clf.predict_proba(F)[:, 1]

    def importance(self):
        return sorted(zip(FEATURE_NAMES, self.clf.feature_importances_), key=lambda x: -x[1])


try:
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            def block(a, b):
                return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
            self.conv = nn.Sequential(block(1, 8), block(8, 16), block(16, 32), nn.AdaptiveAvgPool2d(1))
            self.head = nn.Sequential(nn.Linear(32 + 2, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

        def forward(self, m, rel):
            z = self.conv(m).flatten(1)
            return self.head(torch.cat([z, rel], dim=1)).squeeze(1)

    HAS_TORCH = True
except ImportError:  # 파이토치가 없으면 CNN만 건너뜀
    HAS_TORCH = False


class CNNModel:
    name = "CNN (딥러닝)"

    def __init__(self, epochs=15, seed=0):
        if not HAS_TORCH:
            raise RuntimeError("CNN을 쓰려면 pip install torch")
        torch.manual_seed(seed)
        self.epochs = epochs
        self.net = _Net()

    @staticmethod
    def _prep(M, F):
        M = M[:, :, : M.shape[2] // 2 * 2].reshape(M.shape[0], M.shape[1], -1, 2).mean(axis=3)   # 시간축 절반 (속도)
        m = (M - M.mean(axis=(1, 2), keepdims=True)) / (M.std(axis=(1, 2), keepdims=True) + 1e-6)
        rel = np.clip(F[:, [I_REL, I_PEAK]], -10, 60) / 20     # 소리 크기 정보 (정규화로 사라진 부분)
        return torch.tensor(m[:, None], dtype=torch.float32), torch.tensor(rel, dtype=torch.float32)

    def fit(self, F, M, y, w=None):
        m, rel = self._prep(M, F)
        yt = torch.tensor(y, dtype=torch.float32)
        wt = torch.tensor(w if w is not None else np.ones(len(y)), dtype=torch.float32)
        pos = max(float(y.mean()), 1e-3)
        wt = wt * torch.where(yt > 0, 0.5 / pos, 0.5 / (1 - pos))     # 클래스 균형
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        n = len(y)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = (loss_fn(self.net(m[b], rel[b]), yt[b]) * wt[b]).mean()
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, F, M):
        m, rel = self._prep(M, F)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(m), 256):
                out.append(torch.sigmoid(self.net(m[i:i + 256], rel[i:i + 256])).numpy())
        return np.concatenate(out) if out else np.zeros(0)


def make_model(kind, seed=0):
    return {"rule": RuleModel, "rf": ForestModel, "cnn": CNNModel}[kind](**({} if kind == "rule" else {"seed": seed}))
