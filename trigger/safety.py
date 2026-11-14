"""자극 요청 안전 게이트 (소프트웨어 쪽)

탐지기가 "코골이 확정" / "이갈이 에피소드" 를 내면 바로 자극하지 않고, 아래 조건을 모두 통과해야 '요청'을 보낸다.
모든 판단(통과 · 보류)은 CSV로 남긴다 → 나중에 "언제, 왜 자극했는지 / 안 했는지" 설명 가능.

  ① 무장(arm)       : 기본은 '기록만(marker)'. 실제 장치로 보내려면 명시적으로 --arm 을 켜야 함
  ② 시작 유예        : 측정 시작 후 N분은 요청 안 함 (잠드는 중, 전극 안정화)
  ③ 확신도          : 탐지 확률/세기가 기준 이상일 때만
  ④ 신호 품질        : 마이크 클리핑, 근전도 포화가 있으면 보류 (잘못된 신호로 자극하지 않기)
  ⑤ 최소 간격        : 요청 사이 최소 N초
  ⑥ 시간당 / 밤당 최대 횟수
  ⑦ 무반응 정지      : 연속 N번 요청했는데 코골이가 계속되면 일정 시간 쉼 (같은 사람에게 반복 자극 방지)
  ⑧ 비상 정지        : STOP 파일이 생기거나 stop() 이 호출되면 그 뒤로 전부 보류

세기 · 파형 · 지속시간은 이 소프트웨어가 정하지 않는다. PC는 "요청" 만 보내고,
실제 출력은 하드웨어 쪽(승인된 장치, 자체 제한 회로)이 책임진다.
"""
import csv
import os
import time
from dataclasses import dataclass, field


@dataclass
class SafetyConfig:
    armed: bool = False             # False = 기록만 (가상 자극 마커)
    warmup_min: float = 20.0        # 시작 유예 (분)
    min_conf: float = 0.7           # 확신도 기준
    gap_s: float = 60.0             # 요청 최소 간격
    max_per_hour: int = 20
    max_per_night: int = 80
    nonresponse_n: int = 3          # 연속 요청 후에도 계속되면
    nonresponse_pause_min: float = 30.0   # 이만큼 쉼
    response_window_s: float = 30.0       # 요청 뒤 이 시간 안에 같은 종류가 다시 나오면 '무반응'
    stop_file: str = "STOP"


@dataclass
class SafetyGate:
    cfg: SafetyConfig = field(default_factory=SafetyConfig)
    log_path: str = None
    on_send: callable = None        # 통과했을 때 호출 (출력 장치)
    log: list = field(default_factory=list)
    _sent: list = field(default_factory=list)
    _streak: dict = field(default_factory=dict)
    _pause_until: float = -1.0
    _stopped: bool = False

    def stop(self, reason="수동 정지"):
        self._stopped = True
        self._write(0.0, "정지", reason, "", 0.0, {})

    def request(self, t, source, conf, quality_ok=True, info=None):
        """t: 측정 시작 후 초, source: SNORE/BRUX, conf: 0~1. 반환 True = 요청 보냄"""
        info = info or {}
        reason = self._check(t, source, conf, quality_ok)
        if reason:
            self._write(t, "보류", reason, source, conf, info)
            return False
        self._sent.append(t)
        prev = self._streak.get(source)
        if prev and t - prev[0] <= self.cfg.gap_s + self.cfg.response_window_s:
            self._streak[source] = (t, prev[1] + 1)
        else:
            self._streak[source] = (t, 1)
        if self._streak[source][1] >= self.cfg.nonresponse_n:
            self._pause_until = t + self.cfg.nonresponse_pause_min * 60
            self._streak[source] = (t, 0)
        action = "요청" if self.cfg.armed else "마커"
        self._write(t, action, "", source, conf, info)
        if self.on_send:
            self.on_send(t, source, conf, armed=self.cfg.armed)
        return True

    def _check(self, t, source, conf, quality_ok):
        c = self.cfg
        if self._stopped or (c.stop_file and os.path.exists(c.stop_file)):
            self._stopped = True
            return "비상 정지 상태"
        if t < c.warmup_min * 60:
            return f"시작 유예 {c.warmup_min:.0f}분"
        if conf < c.min_conf:
            return f"확신도 {conf:.2f} < {c.min_conf}"
        if not quality_ok:
            return "신호 품질 불량(클리핑/포화)"
        if t < self._pause_until:
            return f"무반응 {c.nonresponse_n}회 → {c.nonresponse_pause_min:.0f}분 쉬는 중"
        if self._sent and t - self._sent[-1] < c.gap_s:
            return f"최소 간격 {c.gap_s:.0f}초"
        if sum(1 for x in self._sent if t - x < 3600) >= c.max_per_hour:
            return f"시간당 최대 {c.max_per_hour}회"
        if len(self._sent) >= c.max_per_night:
            return f"밤당 최대 {c.max_per_night}회"
        return ""

    def _write(self, t, action, reason, source, conf, info):
        row = {"wall": time.strftime("%Y-%m-%d %H:%M:%S"), "t_s": round(t, 2), "source": source,
               "action": action, "reason": reason, "conf": round(float(conf), 3),
               "info": ";".join(f"{k}={v}" for k, v in info.items())}
        self.log.append(row)
        if self.log_path:
            new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)

    def summary(self):
        n = lambda a: sum(1 for r in self.log if r["action"] == a)
        return {"요청": n("요청"), "마커": n("마커"), "보류": n("보류")}
