"""휴대폰 녹음을 폴더째 넣기 (파일 이름으로 참가자와 소리 종류를 알아서 정리)

    python tools/import_folder.py 녹음폴더
    python tools/import_folder.py 녹음폴더 --subject P01        # 폴더 전체가 한 사람이면
    python tools/import_folder.py 녹음폴더 --dry-run            # 넣지 않고 어떻게 분류되는지만 보기

파일 이름 규칙 (대충 써도 됨)
  P01_코골이_1.m4a        → 참가자 P01, 코골이
  P01_이불코골이.m4a      → 참가자 P01, 이불 덮고 코골이   (이불 + 코골 이 같이 있으면)
  P02_옷소리.m4a          → 옷·이불 소리                    (이불만 있으면 여기)
  P02_뒤척임.wav / P02_말소리.wav / P02_조용함.wav
  P03_밤녹음.m4a + P03_밤녹음.txt  → 라벨(.txt)이 있으면 '긴 녹음'으로 data/recordings 에

  참가자 번호가 파일 이름에 없으면 폴더 이름(예: 녹음폴더/P01/코골이.m4a)이나 --subject 를 씀
  영어도 됨: snore, covered, cloth, movement, speech, quiet

지원 형식: wav, m4a, mp3, aac, ogg, flac 등 (m4a 는 ffmpeg 필요: pip install imageio-ffmpeg)
"""
import argparse
import os
import re
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snore.config import LABEL_KO, SR  # noqa: E402
from snore.dataio import write_audacity_labels  # noqa: E402
from tools.import_data import COMPRESSED, read_audio, read_marks  # noqa: E402

AUDIO_EXT = (".wav",) + COMPRESSED
WORDS = {   # 순서 중요: 위에서부터 먼저 맞는 것
    "snore_covered": [("코골", "이불"), ("snore", "cover"), ("snore", "blanket"), ("코골", "덮")],
    "snore": [("코골",), ("snore",), ("코고",)],
    "cloth": [("옷",), ("이불",), ("cloth",), ("rustl",), ("blanket",)],
    "movement": [("뒤척",), ("움직",), ("move",), ("turn",)],
    "speech": [("말",), ("잠꼬대",), ("speech",), ("talk",), ("voice",)],
    "quiet": [("조용",), ("무음",), ("quiet",), ("silen",)],
}


def guess_label(name):
    n = name.lower().replace(" ", "")
    for lab, rules in WORDS.items():
        if any(all(w in n for w in rule) for rule in rules):
            return lab
    return None


def guess_subject(path, root, default):
    m = re.search(r"(?<![A-Za-z])[pP]\d{1,3}", os.path.relpath(path, root).replace(os.sep, " ").replace("_", " "))
    if m:
        return m.group(0).upper()
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--subject", help="참가자 번호를 파일 이름에서 못 찾을 때 쓸 값")
    ap.add_argument("--data", default="data")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan, skipped = [], []
    for dp, _, files in os.walk(a.folder):
        for fn in sorted(files):
            if not fn.lower().endswith(AUDIO_EXT):
                continue
            p = os.path.join(dp, fn)
            stem = os.path.splitext(fn)[0]
            subj = guess_subject(p, a.folder, a.subject)
            marks = next((os.path.join(dp, stem + e) for e in (".txt", ".csv") if os.path.exists(os.path.join(dp, stem + e))), None)
            lab = None if marks else guess_label(stem)
            if not subj:
                skipped.append((p, "참가자 번호 없음 (--subject P01 처럼 지정)"))
            elif not marks and not lab:
                skipped.append((p, "소리 종류를 모름 (파일 이름에 코골이/이불/옷/뒤척임/말소리/조용함)"))
            else:
                plan.append((p, subj, lab, marks))

    print(f"찾은 녹음 {len(plan) + len(skipped)}개 · 넣을 것 {len(plan)}개 · 건너뜀 {len(skipped)}개\n")
    for p, subj, lab, marks in plan:
        what = f"긴 녹음 + 라벨({os.path.basename(marks)})" if marks else LABEL_KO.get(lab, lab)
        print(f"  {subj}  {what:16s}  ← {os.path.relpath(p, a.folder)}")
    for p, why in skipped:
        print(f"  건너뜀: {os.path.relpath(p, a.folder)}  ({why})")
    if a.dry_run or not plan:
        return

    print()
    for p, subj, lab, marks in plan:
        x, _ = read_audio(p)
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        stem = re.sub(r"[^\w가-힣-]+", "_", os.path.splitext(os.path.basename(p))[0])
        if marks:
            d = os.path.join(a.data, "recordings")
            os.makedirs(d, exist_ok=True)
            base = os.path.join(d, stem if stem.upper().startswith(subj) else f"{subj}_{stem}")
            wavfile.write(base + ".wav", SR, pcm)
            dur = len(x) / SR
            write_audacity_labels(base + ".txt", [(max(0, s), min(dur, e), l) for s, e, l in read_marks(marks, dur) if e > s])
            print(f"저장 {base}.wav (+ 라벨)  {dur:.0f}초")
        else:
            d = os.path.join(a.data, "subjects", subj, lab)
            os.makedirs(d, exist_ok=True)
            out = os.path.join(d, stem + ".wav")
            wavfile.write(out, SR, pcm)
            print(f"저장 {out}  {len(x) / SR:.0f}초")
    print("\n다음: python tools/check_data.py  →  python train.py")


if __name__ == "__main__":
    main()
