"""PC → ESP32 자극 요청 전송

규약 (한 줄 텍스트, 끝에 \\n)
  PC  → ESP  TRG,<번호>,<SNORE|BRUX>,<확신도 0~100>,<단계 1~3>*<CRC8 16진수>   자극 요청
             (단계 번호만 보냄. 단계별 진동 패턴·최대 세기는 ESP32 쪽에 고정)
  PC  → ESP  HB,<번호>*<CRC8>                                       1초마다 살아 있음 신호
  PC  → ESP  STOP*<CRC8>                                            즉시 정지 (이후 요청 무시)
  ESP → PC   ACK,<번호>  /  NAK,<번호>,<이유>

ESP32 쪽 규칙 (firmware/trigger_receiver 참고)
  - HB가 3초 넘게 안 오면 자동 정지 (PC 프로그램이 죽어도 계속 동작하지 않게)
  - CRC 틀리면 무시, 자체 최소 간격 · 시간당 최대 횟수도 따로 검사 (PC를 믿지 않음)
  - 세기 · 시간은 ESP32/하드웨어에 고정. PC는 바꿀 수 없음

출력 종류
  MarkerOutput : 기록만 (기본, 가상 자극 마커)
  SerialOutput : ESP32로 전송 (--arm 일 때만 실제 요청, 아니면 기록만)
"""
import threading
import time


def crc8(data: bytes) -> int:
    """CRC-8 (다항식 0x07)"""
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def frame(body: str) -> bytes:
    return f"{body}*{crc8(body.encode()):02X}\n".encode()


def parse(line: bytes):
    """ESP32 쪽에서 쓰는 것과 같은 검사 (시험용)"""
    s = line.decode(errors="ignore").strip()
    if "*" not in s:
        return None
    body, c = s.rsplit("*", 1)
    try:
        return body if int(c, 16) == crc8(body.encode()) else None
    except ValueError:
        return None


class MarkerOutput:
    def __init__(self, printer=print):
        self.p = printer

    def __call__(self, t, source, conf, armed=False, level=1):
        self.p(f"    ◆ [가상 자극 마커] {source} t={t:.1f}s 확신도 {conf:.2f} 단계 {level}")

    def close(self):
        pass


class SerialOutput:
    def __init__(self, port, baud=115200, printer=print, ser=None, ack_timeout=0.5):
        if ser is None:
            import serial
            ser = serial.Serial(port, baud, timeout=0.05)
        self.ser, self.p, self.seq, self.ack_timeout = ser, printer, 0, ack_timeout
        self._lock = threading.Lock()
        self._run = True
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    def _send(self, body):
        with self._lock:
            self.ser.write(frame(body))

    def _heartbeat(self):
        while self._run:
            self.seq += 1
            try:
                self._send(f"HB,{self.seq}")
            except Exception as e:  # 연결 끊김
                self.p(f"    ! 심장박동 전송 실패: {e}")
            time.sleep(1.0)

    def __call__(self, t, source, conf, armed=False, level=1):
        if not armed:
            self.p(f"    ◆ [기록만] {source} t={t:.1f}s (무장 안 됨 → ESP32로 보내지 않음)")
            return False
        self.seq += 1
        n = self.seq
        self._send(f"TRG,{n},{source},{int(round(conf * 100))},{int(level)}")
        end = time.time() + self.ack_timeout
        while time.time() < end:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line.startswith(f"ACK,{n}"):
                self.p(f"    ▶ 요청 #{n} {source} {level}단계 → ESP32 수락")
                return True
            if line.startswith(f"NAK,{n}"):
                self.p(f"    ✕ 요청 #{n} 거절: {line}")
                return False
        self.p(f"    ? 요청 #{n} 응답 없음 (ESP32 확인)")
        return False

    def close(self):
        self._run = False
        try:
            self._send("STOP")
            self.ser.close()
        except Exception:
            pass
