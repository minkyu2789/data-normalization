"""자극 알고리즘 (진동 피드백): 몇 단계로, 언제 세게, 효과는 있었나

안전 게이트를 통과한 요청마다 '단계'를 정한다.
  1단계: 짧고 약하게      (처음 요청)
  2단계: 조금 길고 세게   (같은 종류가 90초 안에 다시 요청됨 = 1단계로 안 멈춤)
  3단계: 가장 길고 세게   (2단계로도 안 멈춤) — 더 올라가지 않음
  90초 넘게 조용했다가 다시 시작하면 1단계부터.
  3번 연속 요청에도 계속되면 안전 게이트가 30분 쉼 (safety.py)

단계별 진동 패턴은 ESP32 쪽에 고정 (firmware/trigger_receiver). PC는 '단계 번호'만 보냄.

효과 확인: 요청 뒤 30초 동안 같은 종류(코골이 소리 / 이갈이)가 다시 나오는지 기록
  → 밤이 끝나면 "자극 후 멈춘 비율", "단계별 멈춘 비율" 요약
"""
from dataclasses import dataclass, field

PATTERNS = {   # 표시용 (실제 값은 펌웨어에 고정)
    1: "진동 0.2초 × 2번, 약하게",
    2: "진동 0.3초 × 3번, 중간",
    3: "진동 0.4초 × 5번, 세게",
}


@dataclass
class StimPolicy:
    output: callable                    # MarkerOutput / SerialOutput
    escalate_s: float = 90.0
    response_s: float = 30.0
    max_level: int = 3
    printer: callable = print
    history: list = field(default_factory=list)   # [{t, source, level, stopped}]
    _last: dict = field(default_factory=dict)

    def __call__(self, t, source, conf, armed=False):
        prev = self._last.get(source)
        level = min(prev[1] + 1, self.max_level) if prev and t - prev[0] <= self.escalate_s else 1
        self._last[source] = (t, level)
        self.history.append({"t": t, "source": source, "level": level, "stopped": None, "last_seen": t})
        self.printer(f"    ◆ 자극 {level}단계 ({PATTERNS[level]})")
        return self.output(t, source, conf, armed=armed, level=level)

    def observe(self, t, source):
        """탐지기가 소리/버스트를 볼 때마다 호출 → 효과 판정에 사용"""
        for h in self.history:
            if h["source"] == source and h["stopped"] is None and h["t"] < t <= h["t"] + self.response_s:
                h["stopped"] = False

    def close(self, t_end):
        for h in self.history:
            if h["stopped"] is None and t_end >= h["t"] + self.response_s:
                h["stopped"] = True

    def summary(self):
        done = [h for h in self.history if h["stopped"] is not None]
        out = {"자극 횟수": len(self.history),
               "30초 안에 멈춘 비율": round(sum(h["stopped"] for h in done) / len(done), 2) if done else None}
        for lv in (1, 2, 3):
            d = [h for h in done if h["level"] == lv]
            if d:
                out[f"{lv}단계 멈춘 비율"] = f"{sum(h['stopped'] for h in d)}/{len(d)}"
        return out
