"""이갈이(턱 근전도, EMG) 전처리 · 정규화 · 탐지 · 지표

입력: 관자놀이(측두근)나 턱(교근) 위 전극 1채널, µV. 기본 1000 SPS (ADS1292R, 동기 장치 emg.npy)

[전처리]
  1. 직류 제거
  2. 노치 60 / 120 / 180 Hz        한국 전원 잡음과 배수
  3. 대역통과 20–400 Hz           20 Hz 아래 = 움직임·땀·전극 흔들림 (샘플링이 낮으면 상한 자동 조정)
  4. 신호 품질 검사                포화(입력 한계), 값 멈춤, 전원 잡음 비율 → 판단 제외 구간
  5. 포락선                        100 ms RMS, 10 ms 간격 = 근육 활동 '세기'

[정규화]  사람·전극 위치·피부 상태마다 크기가 다르므로
  - MVC 기준 (권장)  : 깨어 있을 때 최대로 꽉 문 힘 = 100%. 기준 10 %MVC (Lavigne 계열 연구는 10–20%)
  - 배경 기준 (AASM) : 최근 60초 '조용한 근육' 크기의 2배. 밤새 바뀌는 배경을 따라감
  - 둘 다 있으면 둘 중 '높은' 기준을 써서 오탐을 줄임

[탐지]  AASM 수면 판독 기준 + Lavigne 1996 기준을 따름
  - 버스트   : 기준을 0.25초 이상 넘는 활동 (0.1초 이내 끊김은 하나로)
  - 위상성   : 0.25–2초 버스트가 3번 이상 규칙적으로 (이 갈기, 딱딱 부딪힘)
  - 긴장성   : 2초 넘는 버스트 (꽉 물기)
  - 혼합     : 둘 다
  - 에피소드 : 버스트 사이 3초 넘게 쉬면 나눔. 한 번짜리 버스트(삼키기, 기침)는 제외

[지표]  시간당 에피소드 수, 시간당 버스트 수, 종류별 개수, 평균 길이
  참고 기준(Lavigne 1996, 연구용): 시간당 에피소드 4회 초과 또는 시간당 버스트 25회 초과 = 빈도 높음
  ※ 진단이 아님. 실제 판정은 수면다원검사와 전문의
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal

FS = 1000
NOTCH = (60, 120, 180)
BAND = (20, 400)
ENV_WIN_S = 0.10
ENV_STEP_S = 0.01
VREF, GAIN = 2.42, 6
SAT_UV = VREF / GAIN * 1e6 * 0.98          # 최대 입력 ±403 mV(게인 6)의 98% 이상이면 포화

BURST_MIN_S = 0.25
PHASIC_MAX_S = 2.0
EPISODE_GAP_S = 3.0
PHASIC_MIN_BURSTS = 3
MVC_THR = 0.10            # 10 %MVC
BG_MULT = 2.0             # AASM: 배경의 2배
BG_WIN_S = 60.0           # 배경 = 최근 60초
HIGH_EP_H, HIGH_BURST_H = 4.0, 25.0


def _band(fs):
    return BAND[0], min(BAND[1], fs / 2 * 0.9)


def quality(x, fs=FS):
    """원신호 품질: 포화·멈춤 표시 (샘플 단위), 전원 잡음 비율"""
    x = np.asarray(x, dtype=np.float64)
    sat = np.abs(x) >= SAT_UV
    flat = np.r_[False, np.abs(np.diff(x)) < 1e-6]
    flat = np.convolve(flat, np.ones(20), "same") >= 20
    f, p = signal.welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4 * int(fs)))
    lo, hi = _band(fs)
    emg_p = p[(f >= lo) & (f <= hi)].sum() + 1e-12
    hum_p = sum(p[(f > h - 2) & (f < h + 2)].sum() for h in NOTCH if h < fs / 2)
    return sat | flat, float(hum_p / emg_p)


def clean(x, fs=FS):
    """µV 원신호 → 필터된 신호, 나쁜 샘플 표시"""
    x = np.asarray(x, dtype=np.float64)
    bad, _ = quality(x, fs)
    y = x - np.median(x)
    for f0 in NOTCH:
        if f0 < fs / 2:
            b, a = signal.iirnotch(f0, 30, fs)
            y = signal.filtfilt(b, a, y)
    sos = signal.butter(4, _band(fs), btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, y), bad


def envelope(y, fs=FS, win_s=ENV_WIN_S, step_s=ENV_STEP_S):
    """RMS 포락선 → (시각, 세기 µV)"""
    w, h = max(1, int(win_s * fs)), max(1, int(step_s * fs))
    p = np.convolve(y ** 2, np.ones(w) / w, "same")
    idx = np.arange(0, len(y), h)
    return idx / fs, np.sqrt(p[idx])


def background(env, step_s=ENV_STEP_S, win_s=BG_WIN_S):
    """움직이는 배경 크기: 최근 win_s 동안 포락선의 하위 30% 값 (버스트에 끌려가지 않게)"""
    n = max(1, int(win_s / step_s))
    k = max(1, n // 50)                         # 계산을 줄이려고 50칸마다
    pts = np.arange(0, len(env), k)
    bg = np.array([np.percentile(env[max(0, i - n):i + 1], 30) for i in pts])
    return np.interp(np.arange(len(env)), pts, bg)


def mvc_from_calibration(x, fs=FS, segments=None):
    """꽉 물기 보정 → MVC (µV). segments=[(시작,끝)] 이 있으면 그 안에서, 없으면 가장 센 1초"""
    y, _ = clean(x, fs)
    t, env = envelope(y, fs)
    k = max(1, int(1.0 / ENV_STEP_S))
    if segments:
        vals = []
        for s, e in segments:
            m = env[(t >= s) & (t <= e)]
            if len(m) >= k:
                vals.append(np.convolve(m, np.ones(k) / k, "valid").max())
        if vals:
            return float(np.median(vals))
    if len(env) < k:
        return float(env.max())
    return float(np.convolve(env, np.ones(k) / k, "valid").max())


@dataclass
class Norm:
    mvc_uv: float = None
    mvc_thr: float = MVC_THR
    bg_mult: float = BG_MULT

    def threshold(self, bg):
        """샘플마다 기준 (µV): MVC 기준과 배경 기준 중 높은 것"""
        thr = self.bg_mult * bg
        if self.mvc_uv:
            thr = np.maximum(thr, self.mvc_thr * self.mvc_uv)
        return thr

    def text(self):
        s = f"배경의 {self.bg_mult:.0f}배"
        if self.mvc_uv:
            s = f"MVC {self.mvc_uv:.1f} µV의 {self.mvc_thr * 100:.0f}% 와 " + s + " 중 높은 값"
        return s


def bursts(t, env, thr, bad=None, min_s=BURST_MIN_S, merge_s=0.1):
    """기준 이상 구간 → [(시작, 끝, 기준 대비 최대 배수)]"""
    on = env >= thr
    if bad is not None:
        on &= ~bad
    d = np.diff(np.r_[0, on.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    segs = []
    for s, e in zip(st, en):
        pk = float((env[s:e + 1] / thr[s:e + 1]).max())
        if segs and t[s] - segs[-1][1] <= merge_s:
            ps, _, pp = segs[-1]
            segs[-1] = (ps, t[e], max(pp, pk))
        else:
            segs.append((t[s], t[e], pk))
    return [s for s in segs if s[1] - s[0] >= min_s]


def classify(bs):
    """버스트 → 에피소드 [(시작, 끝, 종류, 버스트 수, 최대 배수)]"""
    eps, grp = [], []
    for b in bs:
        if grp and b[0] - grp[-1][1] > EPISODE_GAP_S:
            eps.append(_ep(grp))
            grp = []
        grp.append(b)
    if grp:
        eps.append(_ep(grp))
    return [e for e in eps if e is not None]


def _ep(grp):
    durs = [e - s for s, e, _ in grp]
    tonic = any(d > PHASIC_MAX_S for d in durs)
    phasic = sum(BURST_MIN_S <= d <= PHASIC_MAX_S for d in durs) >= PHASIC_MIN_BURSTS
    if not (tonic or phasic):
        return None
    kind = "혼합" if tonic and phasic else ("긴장성" if tonic else "위상성")
    return (grp[0][0], grp[-1][1], kind, len(grp), max(p for *_, p in grp))


def indices(eps, bs, hours):
    h = max(hours, 1e-9)
    kinds = {k: sum(1 for e in eps if e[2] == k) for k in ("위상성", "긴장성", "혼합")}
    ep_h, b_h = len(eps) / h, len(bs) / h
    return {
        "기록 시간(시간)": round(hours, 2),
        "에피소드 수": len(eps), "시간당 에피소드": round(ep_h, 1),
        "버스트 수": len(bs), "시간당 버스트": round(b_h, 1),
        "위상성": kinds["위상성"], "긴장성": kinds["긴장성"], "혼합": kinds["혼합"],
        "에피소드 평균 길이(초)": round(float(np.mean([e[1] - e[0] for e in eps])), 1) if eps else 0.0,
        "빈도 참고(연구 기준)": "높음" if (ep_h > HIGH_EP_H or b_h > HIGH_BURST_H) else "낮음",
    }


def analyze(x, fs=FS, mvc_uv=None, norm=None):
    y, bad_s = clean(x, fs)
    t, env = envelope(y, fs)
    idx = (t * fs).astype(int).clip(0, len(bad_s) - 1)
    bad = bad_s[idx]
    bg = background(env)
    norm = norm or Norm(mvc_uv=mvc_uv)
    thr = norm.threshold(bg)
    bs = bursts(t, env, thr, bad)
    eps = classify(bs)
    _, hum = quality(x, fs)
    hours = len(x) / fs / 3600
    return {"y": y, "t": t, "env": env, "bg": bg, "thr": thr, "bad": bad, "norm": norm,
            "bursts": bs, "episodes": eps, "metrics": indices(eps, bs, hours),
            "index": len(eps) / hours if hours > 0 else 0.0,
            "sat_ratio": float(bad_s.mean()), "hum_ratio": hum}
