"""후처리 + 트리거

창별 확률 (0.5초마다)
 → ① 중앙값 필터로 흔들림 제거
 → ② 이어진 코골이 창을 하나의 '소리'로 묶음
 → ③ 소리가 호흡 간격(1.5–10초)으로 3번 이상 반복되면 '코골이 확정'
 → ④ 트리거 판단 (최소 간격, 시간당 최대 횟수) → on_trigger 호출 + 기록

①–③ 덕분에 옷 소리나 기침처럼 한 번 나는 소리는 확률이 높아도 '확정'되지 않음
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from .config import EPISODE_MIN, HOP_S, INTERVAL, PROB_THR, SMOOTH, TRIGGER_GAP_S, TRIGGER_MAX_H, WIN_S


def smooth(prob, k=SMOOTH):
    return median_filter(prob, size=k, mode="nearest") if len(prob) else prob


def to_events(times, flags, prob=None):
    """연속된 True 창 → (시작, 끝, 최대 확률)"""
    ev, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if (not f or i == len(flags) - 1) and start is not None:
            end = i if f else i - 1
            pk = float(prob[start:end + 1].max()) if prob is not None else 1.0
            ev.append((float(times[start]), float(times[end] + WIN_S), pk))
            start = None
    return ev


def episodes(events):
    """호흡 간격으로 반복되는 소리 묶음 → 코골이 구간 (시작, 끝, 횟수)"""
    out, grp = [], []
    for e in events:
        if grp and INTERVAL[0] <= e[0] - grp[-1][0] <= INTERVAL[1]:
            grp.append(e)
        else:
            if len(grp) >= EPISODE_MIN:
                out.append((grp[0][0], grp[-1][1], len(grp)))
            grp = [e]
    if len(grp) >= EPISODE_MIN:
        out.append((grp[0][0], grp[-1][1], len(grp)))
    return out


@dataclass
class TriggerController:
    """코골이 확정 시 트리거 신호를 내보낼지 판단.
    on_trigger(t, info) 에 하드웨어 쪽 함수를 연결 (기본은 기록만).
    하드웨어 쪽에도 별도의 세기·시간 제한이 반드시 있어야 함."""
    gap_s: float = TRIGGER_GAP_S
    max_per_hour: int = TRIGGER_MAX_H
    on_trigger: callable = None
    log: list = field(default_factory=list)
    _times: list = field(default_factory=list)

    def request(self, t, info):
        recent = [x for x in self._times if t - x < 3600]
        if self._times and t - self._times[-1] < self.gap_s:
            self.log.append({"t": t, "action": "보류", "reason": f"최소 간격 {self.gap_s:.0f}초", **info})
            return False
        if len(recent) >= self.max_per_hour:
            self.log.append({"t": t, "action": "보류", "reason": f"시간당 최대 {self.max_per_hour}회", **info})
            return False
        self._times.append(t)
        self.log.append({"t": t, "action": "트리거", "reason": "", **info})
        if self.on_trigger:
            self.on_trigger(t, info)
        return True


class StreamDetector:
    """0.5초마다 확률이 하나씩 들어오는 실시간 상황을 흉내 (콘솔 출력용)"""

    def __init__(self, controller=None, printer=print, repeat_s=None):
        self.ctl = controller or TriggerController()
        self.repeat_s = repeat_s      # 코골이가 계속되면 이 간격마다 다시 요청 (None = 에피소드당 1번)
        self.fire_t = None
        self.retry_t = None
        self.ep_peaks = []
        self.p = printer
        self.buf = []
        self.in_sound = None
        self.sounds = []
        self.ep_count = 0
        self.ep_fired = False

    def push(self, t, prob):
        self.buf.append(prob)
        sp = float(np.median(self.buf[-SMOOTH:]))
        on = sp >= PROB_THR
        if on and self.in_sound is None:
            self.in_sound = (t, prob)
        elif not on and self.in_sound is not None:
            s, pk = self.in_sound
            self.in_sound = None
            self._sound(s, t - HOP_S + WIN_S, max(pk, prob))
        elif on:
            self.in_sound = (self.in_sound[0], max(self.in_sound[1], prob))

    def _sound(self, s, e, pk):
        if self.sounds and INTERVAL[0] <= s - self.sounds[-1][0] <= INTERVAL[1]:
            self.ep_count += 1
            self.ep_peaks.append(pk)
        else:
            self.ep_count = 1
            self.ep_peaks = [pk]
            self.ep_fired = False
            self.retry_t = None
        self.sounds.append((s, e))
        self.p(f"[{_hms(s)}] 코골이 후보 소리   길이 {e - s:4.1f}초  확률 {pk:.2f}  (연속 {self.ep_count}회)")
        if self.ep_count < EPISODE_MIN:
            return
        again = (self.ep_fired and self.repeat_s is not None and self.fire_t is not None
                 and e - self.fire_t >= self.repeat_s)
        retry = not self.ep_fired and (self.retry_t is None or e >= self.retry_t)
        if retry or again:
            conf = float(np.mean(self.ep_peaks[-EPISODE_MIN:]))      # 최근 3번 소리의 평균 확률
            ok = self.ctl.request(e, {"repeats": self.ep_count, "prob": round(conf, 2)})
            last = self.ctl.log[-1]
            if ok:
                self.ep_fired, self.fire_t = True, e
                self.p(f"[{_hms(e)}] ▶ 코골이 확정 ({self.ep_count}회 반복) → 트리거 #{len(self.ctl._times)}")
            else:
                self.retry_t = e + 10          # 보류되면 10초 뒤 다시 판단 (기록이 너무 많아지지 않게)
                if again:
                    self.fire_t = e
                self.p(f"[{_hms(e)}]   코골이 확정 → 트리거 보류 ({last['reason']})")


def _hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
