"""실시간: 마이크 코골이 + 턱 근전도 이갈이 → 안전 게이트 → 자극 알고리즘(단계별 진동) → ESP32

  # 녹음 파일을 실제 시간처럼 흘려 보내기 (장치 없이 흐름 확인, --speed 0 = 최대 속도)
  python live.py --wav data/test/P99_night.wav --speed 0 --warmup 0

  # 소리 + 근전도 같이 (동기 장치 세션 폴더)
  python live.py --wav 세션/audio.wav --emg 세션/emg.npy --calib results/brux_calib_P01.json --speed 0 --warmup 0

  # 근전도만
  python live.py --emg 세션/emg.npy --speed 0 --warmup 0

  # PC 마이크 실시간 (pip install sounddevice)
  python live.py --mic

  # ESP32 로 요청 보내기 (기본은 기록만. --arm 을 붙여야 진동 모터가 실제로 동작)
  python live.py --mic --port COM5 --arm

결과: results/live_<시각>_decisions.csv (통과·보류 판단) + _probs.csv (0.5초마다 코골이 확률) + _stim.csv (자극 단계·효과)
비상 정지: 실행 폴더에 STOP 이라는 파일을 만들거나 Ctrl+C
"""
import argparse
import json
import os
import queue
import sys
import time

import numpy as np
from scipy import signal

from detect import load_model
from snore.config import HOP_S, SR, WIN_S
from snore.features import handcrafted, logmel
from snore.postprocess import StreamDetector
from snore.preprocess import _SOS, frame_db
from trigger.link import MarkerOutput, SerialOutput
from trigger.policy import StimPolicy
from trigger.safety import SafetyConfig, SafetyGate


class LiveFeatures:
    """소리를 조금씩 받아서 0.5초마다 1초 창 특징을 만든다.
    바닥소음은 최근 5분의 하위 10% (방 소리가 바뀌어도 따라감)."""

    def __init__(self, with_mel, floor_db=None, floor_hist_s=300):
        self.w, self.h, self.pad = int(WIN_S * SR), int(HOP_S * SR), int(0.5 * SR)
        self.buf = np.zeros(0, np.float32)
        self.pos = 0              # 다음 창 시작 (전체 샘플 기준)
        self.base = 0             # buf[0] 의 전체 샘플 위치
        self.with_mel = with_mel
        self.fixed_floor = floor_db
        self.hist = []
        self.hist_n = int(floor_hist_s / 0.05)

    def push(self, chunk):
        self.buf = np.concatenate([self.buf, np.asarray(chunk, np.float32)])
        out = []
        while self.pos + self.w <= self.base + len(self.buf):
            a = self.pos - self.base
            lo = max(0, a - self.pad)
            seg = self.buf[lo:a + self.w]
            clip = float(np.mean(np.abs(seg[a - lo:]) > 0.99))
            y = signal.sosfiltfilt(_SOS, seg)[a - lo:] if len(seg) > 64 else seg
            fd = frame_db(y)
            self.hist.extend(fd[: len(fd) // 2].tolist())      # 0.5초 이동이라 절반만 새 값
            self.hist = self.hist[-self.hist_n:]
            floor = self.fixed_floor if self.fixed_floor is not None else float(np.percentile(self.hist, 10))
            rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
            rel = np.array([20 * np.log10(rms) - floor], np.float32)
            peak = np.array([fd.max() - floor], np.float32)
            norm = (y / rms)[None, :].astype(np.float32)
            F = handcrafted(norm, rel, peak)
            M = logmel(norm).astype(np.float32) if self.with_mel else np.zeros(0)
            out.append((self.pos / SR, F, M, clip, floor))
            self.pos += self.h
        keep = self.pos - self.pad - self.base
        if keep > 0:
            self.buf, self.base = self.buf[keep:], self.base + keep
        return out


class GateController:
    """StreamDetector 가 부르는 모양(request/log/_times)에 안전 게이트를 끼움"""

    def __init__(self, gate, quality):
        self.gate, self.quality, self.log, self._times = gate, quality, [], []

    def request(self, t, info):
        ok = self.gate.request(t, "SNORE", info.get("prob", 0.0), self.quality(), info)
        self.log.append({"t": t, "action": "트리거" if ok else "보류", "reason": self.gate.log[-1]["reason"], **info})
        if ok:
            self._times.append(t)
        return ok


def wav_source(path, speed, block=0.1):
    from snore.dataio import load_wav
    x = load_wav(path)
    n = int(block * SR)
    for i in range(0, len(x), n):
        yield x[i:i + n]
        if speed > 0:
            time.sleep(block / speed)


def mic_source(device=None, block=0.1):
    import sounddevice as sd
    q = queue.Queue()
    st = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(block * SR),
                        device=device, callback=lambda d, *_: q.put(d[:, 0].copy()))
    with st:
        while True:
            yield q.get()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--emg", help="근전도 (.npy µV 또는 CSV) — 소리와 같은 시각에 시작한 기록")
    ap.add_argument("--emg-fs", type=float, default=1000)
    ap.add_argument("--calib", help="brux_calibrate.py 결과 json")
    ap.add_argument("--device", help="마이크 장치 번호/이름")
    ap.add_argument("--speed", type=float, default=1.0, help="파일 재생 배속 (0 = 최대)")
    ap.add_argument("--model", default="cnn", choices=["rule", "rf", "cnn"])
    ap.add_argument("--models-dir", default="results")
    ap.add_argument("--port", help="ESP32 시리얼 포트 (없으면 기록만)")
    ap.add_argument("--arm", action="store_true", help="실제 요청 전송 허용")
    ap.add_argument("--warmup", type=float, default=20, help="시작 유예 (분)")
    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--gap", type=float, default=60)
    ap.add_argument("--max-hour", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if not (a.wav or a.mic or a.emg):
        sys.exit("--wav, --mic, --emg 중 하나 이상 필요해요")
    if a.wav and a.mic:
        sys.exit("--wav 와 --mic 는 같이 쓸 수 없어요")
    if a.arm and not a.port:
        sys.exit("--arm 은 --port 와 함께만 쓸 수 있어요")
    if a.arm:
        print("※ 무장 모드: 조건을 통과하면 ESP32 로 진동 요청을 보냅니다.")
        if input("계속하려면 YES 입력: ").strip() != "YES":
            sys.exit("취소")

    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dec_path = os.path.join(a.out, f"live_{stamp}_decisions.csv")
    prob_f = open(os.path.join(a.out, f"live_{stamp}_probs.csv"), "w", encoding="utf-8-sig")
    prob_f.write("t_s,prob,rel_db,peak_db,floor_dbfs,clip\n")

    out = SerialOutput(a.port) if a.port else MarkerOutput()
    policy = StimPolicy(out)
    cfg = SafetyConfig(armed=a.arm, warmup_min=a.warmup, min_conf=a.min_conf, gap_s=a.gap, max_per_hour=a.max_hour)
    gate = SafetyGate(cfg, log_path=dec_path, on_send=policy)

    # 코골이
    has_audio = bool(a.wav or a.mic)
    state = {"clip": 0.0}
    det = lf = model = None
    if has_audio:
        ctl = GateController(gate, quality=lambda: state["clip"] < 0.01)
        det = StreamDetector(ctl, repeat_s=a.gap)
        model = load_model(a.model, a.models_dir)
        lf = LiveFeatures(with_mel=a.model == "cnn")

    # 이갈이
    brux = emg_x = None
    if a.emg:
        from brux.stream import BruxStream
        from brux_detect import read_emg
        emg_x = read_emg(a.emg)
        mvc = json.load(open(a.calib))["mvc_uv"] if a.calib else None

        def brux_request(t, kind, conf):
            if not gate.request(t, "BRUX", conf, True, {"kind": kind}):
                print(f"           이갈이 확정 → 자극 보류 ({gate.log[-1]['reason']})")

        brux = BruxStream(fs=a.emg_fs, mvc_uv=mvc, on_episode=brux_request,
                          on_burst=lambda t: policy.observe(t, "BRUX"))

    print(f"{'코골이(' + a.model + ') ' if has_audio else ''}{'+ 이갈이 ' if brux else ''}"
          f"· 시작 유예 {a.warmup:.0f}분 · 최소 간격 {a.gap:.0f}초 · {'무장' if a.arm else '기록만'} · 비상 정지: STOP 파일 또는 Ctrl+C\n")

    block = 0.1
    if a.mic:
        source = mic_source(a.device, block)
    elif a.wav:
        source = wav_source(a.wav, a.speed, block)
    else:
        n_blk = int(np.ceil(len(emg_x) / (a.emg_fs * block)))
        source = (None for _ in range(n_blk))
    n_snd, k, t_now = 0, 0, 0.0
    try:
        for chunk in source:
            if brux is not None:
                e0, e1 = int(k * block * a.emg_fs), int((k + 1) * block * a.emg_fs)
                if e0 < len(emg_x):
                    brux.push(emg_x[e0:e1])
                if not has_audio and a.speed > 0:
                    time.sleep(block / a.speed)
            k += 1
            t_now = k * block
            if chunk is not None:
                for t, F, M, clip, floor in lf.push(chunk):
                    p = float(model.predict_proba(F, M)[0])
                    state["clip"] = clip
                    prob_f.write(f"{t:.1f},{p:.3f},{F[0, -2]:.1f},{F[0, -1]:.1f},{floor:.1f},{clip:.3f}\n")
                    before = len(det.sounds)
                    det.push(t, p)
                    if len(det.sounds) > before:
                        policy.observe(t, "SNORE")
                    n_snd += 1
            if gate._stopped:
                print("비상 정지됨")
                break
    except KeyboardInterrupt:
        gate.stop("Ctrl+C")
    finally:
        out.close()
        prob_f.close()
    policy.close(t_now)
    with open(os.path.join(a.out, f"live_{stamp}_stim.csv"), "w", encoding="utf-8-sig") as f:
        f.write("t_s,source,level,stopped_within_30s\n")
        for h in policy.history:
            f.write(f"{h['t']:.1f},{h['source']},{h['level']},{h['stopped']}\n")
    s = gate.summary()
    print(f"\n처리 {t_now:.0f}초 · 요청 {s['요청']} · 마커 {s['마커']} · 보류 {s['보류']}")
    print("자극 효과:", ", ".join(f"{k} {v}" for k, v in policy.summary().items()))
    print(f"판단 기록: {dec_path}")


if __name__ == "__main__":
    main()
