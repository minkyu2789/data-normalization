"""실시간 이갈이 탐지 (근전도가 조금씩 들어올 때)

오프라인(emg.py)과 같은 기준을 쓰되, 미래 데이터를 쓰지 않는 필터(인과 필터)로 처리한다.
에피소드가 '끝날 때'까지 기다리지 않고 조건을 만족하는 순간 바로 알림:
  - 위상성: 0.25–2초 버스트가 3번째 나온 순간
  - 긴장성: 버스트가 2초를 넘는 순간
"""
from collections import deque

import numpy as np
from scipy import signal

from . import emg


class BruxStream:
    def __init__(self, fs=emg.FS, mvc_uv=None, on_episode=None, printer=print, on_burst=None):
        self.fs = fs
        self.on_burst = on_burst
        self.norm = emg.Norm(mvc_uv=mvc_uv)
        self.on_episode = on_episode
        self.p = printer
        self.notch = []
        for f0 in emg.NOTCH:
            if f0 < fs / 2:
                b, a = signal.iirnotch(f0, 30, fs)
                self.notch.append((b, a, signal.lfilter_zi(b, a) * 0))
        self.sos = signal.butter(4, emg._band(fs), btype="band", fs=fs, output="sos")
        self.zi = signal.sosfilt_zi(self.sos) * 0
        self.step = max(1, int(emg.ENV_STEP_S * fs))
        self.win = max(1, int(emg.ENV_WIN_S * fs))
        self.sq = deque(maxlen=self.win)
        self.carry = np.zeros(0)
        self.n = 0                                   # 지금까지 받은 샘플 수
        self.hist = deque(maxlen=int(emg.BG_WIN_S / emg.ENV_STEP_S))
        self.bg = None
        self.k = 0
        self.burst_start = None
        self.burst_peak = 0.0
        self.last_on = -1e9
        self.group = []                              # 현재 에피소드 후보의 버스트들
        self.fired = False
        self.dc = None
        self.episodes = []

    def push(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not len(x):
            return
        if self.dc is None:
            self.dc = float(np.median(x))
        bad = (np.abs(x) >= emg.SAT_UV)
        y = x - self.dc
        self.dc = 0.999 * self.dc + 0.001 * float(np.mean(x))
        new = []
        for i, (b, a, zi) in enumerate(self.notch):
            y, zi = signal.lfilter(b, a, y, zi=zi)
            new.append((b, a, zi))
        self.notch = new
        y, self.zi = signal.sosfilt(self.sos, y, zi=self.zi)
        self.carry = np.concatenate([self.carry, y ** 2])
        self._bad = bad.any()
        while len(self.carry) >= self.step:
            blk, self.carry = self.carry[: self.step], self.carry[self.step:]
            self.sq.extend(blk)
            self.n += self.step
            self._tick(self.n / self.fs, float(np.sqrt(np.mean(self.sq))))

    def _tick(self, t, env):
        self.hist.append(env)
        self.k += 1
        if self.bg is None or self.k % 100 == 0:            # 1초마다 배경 갱신
            self.bg = float(np.percentile(self.hist, 30))
        if t < 5:                                           # 처음 5초는 배경만 배움
            return
        thr = float(self.norm.threshold(np.array([self.bg]))[0])
        on = env >= thr and not self._bad
        if on:
            if self.burst_start is None:
                self.burst_start = t
                self.burst_peak = 0.0
            self.burst_peak = max(self.burst_peak, env / thr)
            self.last_on = t
            if t - self.burst_start > emg.PHASIC_MAX_S and not self.fired:
                self._fire(t, "긴장성")
        elif self.burst_start is not None and t - self.last_on > 0.1:
            s, e = self.burst_start, self.last_on
            self.burst_start = None
            if e - s >= emg.BURST_MIN_S:
                if self.group and s - self.group[-1][1] > emg.EPISODE_GAP_S:
                    self.group, self.fired = [], False
                self.group.append((s, e, self.burst_peak))
                if self.on_burst:
                    self.on_burst(e)
                ph = sum(emg.BURST_MIN_S <= b - a <= emg.PHASIC_MAX_S for a, b, _ in self.group)
                if ph >= emg.PHASIC_MIN_BURSTS and not self.fired:
                    self._fire(e, "위상성")
        if self.group and self.burst_start is None and t - self.group[-1][1] > emg.EPISODE_GAP_S:
            self.group, self.fired = [], False

    def _fire(self, t, kind):
        self.fired = True
        peak = max([p for *_, p in self.group] + [self.burst_peak])
        conf = float(np.clip(0.5 + (peak - 1) / 4, 0, 1))
        self.episodes.append((t, kind))
        h = int(t)
        self.p(f"[{h // 3600:02d}:{h % 3600 // 60:02d}:{h % 60:02d}] ▶ 이갈이 확정 ({kind}, 기준의 {peak:.1f}배)")
        if self.on_episode:
            self.on_episode(t, kind, conf)
