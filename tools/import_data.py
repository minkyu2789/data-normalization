"""받은 녹음을 학습용 폴더(data/)로 넣기

어떤 형식이든 16kHz 모노 wav 로 바꿔서 data/recordings/ 또는 data/subjects/ 에 저장한다.

  휴대폰 녹음 (m4a는 먼저 wav로 내보내기)
    python tools/import_data.py phone.wav --subject P01 --label snore
        → data/subjects/P01/snore/phone.wav

  밤새 녹음 + 정답 표시
    python tools/import_data.py night.wav --subject P02 --marks night_marks.csv
        → data/recordings/P02_night.wav + P02_night.txt (Audacity 라벨)

  측정 장치(collect.py 등)에서 저장한 원본 소리
    python tools/import_data.py audio.bin --raw int16 --sr 16000 --subject P03 --marks marks.csv
    python tools/import_data.py audio.npy --sr 16000 --subject P03
    python tools/import_data.py audio.csv --column sample --sr 16000 --subject P03

정답 표시 파일(--marks) 형식 (둘 중 아무거나)
  1) Audacity 라벨     시작<TAB>끝<TAB>라벨
  2) CSV (헤더 있음)    start_s,end_s,label
  3) CSV (누른 순간)    t_s,label   ← 측정 프로그램의 _marks.csv
       같은 라벨이 '시작/끝' 두 번 찍히거나, label 이 'snore_start' / 'snore_end' 처럼 되어 있으면 구간으로 묶음
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snore.config import SR  # noqa: E402
from snore.dataio import write_audacity_labels  # noqa: E402

ALIASES = {  # 사람마다 다르게 적는 라벨을 통일
    "코골이": "snore", "코골이_이불": "snore_covered", "이불코골이": "snore_covered",
    "옷": "cloth", "이불": "cloth", "옷소리": "cloth", "뒤척임": "movement", "움직임": "movement",
    "말": "speech", "말소리": "speech", "조용함": "quiet", "무음": "quiet",
}


def norm_label(s):
    s = s.strip().lower().replace(" ", "_")
    return ALIASES.get(s, s)


COMPRESSED = (".m4a", ".mp3", ".aac", ".ogg", ".flac", ".3gp", ".amr", ".mp4", ".wma", ".opus")


def ffmpeg_exe():
    """ffmpeg 위치: PATH에 있으면 그것, 없으면 pip install imageio-ffmpeg 로 받은 것"""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("m4a/mp3 를 읽으려면:  pip install imageio-ffmpeg")


def decode_compressed(path):
    """휴대폰 녹음(m4a 등) → 16 kHz 모노 float32"""
    import subprocess
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


def read_audio(path, raw=None, sr=None, column=None, channels=1):
    ext = os.path.splitext(path)[1].lower()
    if ext in COMPRESSED:
        x = decode_compressed(path)
        return x - x.mean(), SR
    if ext == ".wav":
        file_sr, x = wavfile.read(path)
    elif ext == ".npy":
        x, file_sr = np.load(path), sr
    elif ext == ".csv":
        with open(path, encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            col = column or [c for c in rd.fieldnames if c.lower() not in ("t", "t_s", "time", "us", "t_us")][0]
            x = np.array([float(r[col]) for r in rd], dtype=np.float64)
        file_sr = sr
    else:                                   # .bin .raw .pcm 등 헤더 없는 원본
        if not raw:
            raise SystemExit("헤더 없는 파일은 --raw int16|int32|float32 와 --sr 를 알려 주세요")
        x = np.fromfile(path, dtype=np.dtype(raw))
        file_sr = sr
    if file_sr is None:
        raise SystemExit("샘플레이트를 모릅니다. --sr 16000 처럼 알려 주세요")
    if x.ndim == 1 and channels > 1:
        x = x[: len(x) // channels * channels].reshape(-1, channels)
    if x.ndim > 1:
        x = x.mean(axis=1)
    # 정수 → -1~1
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    x = np.asarray(x, dtype=np.float64)
    peak = np.abs(x).max() if len(x) else 0
    if peak > 1.0:                          # csv 정수값 등: 최대값 기준으로 맞춤
        bits = 24 if peak > 32768 else 16
        x = x / float(2 ** (bits - 1))
    x = x - x.mean()                        # 직류 성분 제거
    if file_sr != SR:
        from math import gcd
        g = gcd(int(file_sr), SR)
        x = resample_poly(x, SR // g, int(file_sr) // g)
    return np.clip(x, -1, 1).astype(np.float32), int(file_sr)


def read_marks(path, total_s):
    """→ [(시작, 끝, 라벨)]"""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip().splitlines()
    rows = [l for l in text if l.strip() and not l.startswith("\\")]
    if rows and "\t" in rows[0] and not rows[0].lower().startswith("start"):
        out = []
        for l in rows:
            p = l.split("\t")
            out.append((float(p[0]), float(p[1]), norm_label(p[2] if len(p) > 2 else "")))
        return out
    rd = list(csv.DictReader(rows))
    keys = [k.lower() for k in rd[0].keys()] if rd else []
    if "start_s" in keys or "start" in keys:
        g = lambda r, *ks: next(r[k] for k in r if k.lower() in ks)
        return [(float(g(r, "start_s", "start")), float(g(r, "end_s", "end")), norm_label(g(r, "label")))
                for r in rd]
    # 누른 순간 기록 → 구간으로 묶기
    tk = next(k for k in rd[0] if k.lower() in ("t_s", "t", "time"))
    lk = next(k for k in rd[0] if k.lower() in ("label", "mark", "note"))
    pts = sorted((float(r[tk]), r[lk].strip()) for r in rd)
    out, open_ = [], {}
    for t, lab in pts:
        base = lab.lower()
        for suf in ("_start", "_end", " 시작", " 끝", "_시작", "_끝"):
            base = base.replace(suf, "")
        base = norm_label(base)
        ends = lab.lower().endswith(("_end", "끝"))
        if base in open_ and (ends or not lab.lower().endswith(("_start", "시작"))):
            out.append((open_.pop(base), t, base))
        elif not ends:
            open_[base] = t
    for base, s in open_.items():           # 끝 표시를 안 누른 구간은 녹음 끝까지
        out.append((s, total_s, base))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--subject", required=True, help="참가자 번호 (P01 등). 사람마다 꼭 다르게")
    ap.add_argument("--label", help="파일 전체가 한 종류일 때 (snore, cloth, quiet …)")
    ap.add_argument("--marks", help="구간별 정답 파일")
    ap.add_argument("--raw", help="헤더 없는 파일의 자료형: int16 / int32 / float32")
    ap.add_argument("--sr", type=int, help="원본 샘플레이트 (wav는 자동)")
    ap.add_argument("--column", help="csv에서 소리 값이 든 열 이름")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--name", help="저장 이름 (기본: 원본 파일 이름)")
    ap.add_argument("--test", action="store_true", help="학습에 쓰지 않는 시험용(data/test)으로 저장")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    x, src_sr = read_audio(a.src, a.raw, a.sr, a.column, a.channels)
    dur = len(x) / SR
    name = a.name or os.path.splitext(os.path.basename(a.src))[0]
    pcm = (x * 32767).astype(np.int16)
    print(f"읽음: {a.src}  ({src_sr} Hz → {SR} Hz, {dur:.1f}초, 최대 {np.abs(x).max():.2f})")

    if a.label and not a.marks:
        lab = norm_label(a.label)
        d = os.path.join(a.data, "subjects", a.subject, lab)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, name + ".wav")
        wavfile.write(out, SR, pcm)
        print(f"저장: {out}  [라벨 {lab}]")
        return

    d = os.path.join(a.data, "test" if a.test else "recordings")
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"{a.subject}_{name}" if not name.startswith(a.subject) else name)
    wavfile.write(base + ".wav", SR, pcm)
    print(f"저장: {base}.wav")
    if a.marks:
        segs = [(max(0.0, s), min(dur, e), lab) for s, e, lab in read_marks(a.marks, dur) if e > s]
        write_audacity_labels(base + ".txt", segs)
        tot = {}
        for s, e, lab in segs:
            tot[lab] = tot.get(lab, 0) + e - s
        print(f"라벨: {base}.txt  " + ", ".join(f"{k} {v:.0f}초" for k, v in tot.items()))
    else:
        print("※ 정답 표시가 없어서 학습에는 안 쓰여요. Audacity로 라벨을 만들어 같은 이름 .txt 로 저장하세요")


if __name__ == "__main__":
    main()
