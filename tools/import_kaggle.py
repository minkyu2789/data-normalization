"""Kaggle 공개 코골이 데이터셋 넣기 (Khan 2019, "Snoring" — 코골이 500개 + 아닌 소리 500개, 각 1초)

1. https://www.kaggle.com/datasets/tareqkhanemu/snoring 에서 로그인 후 Download → zip 풀기
2. python tools/import_kaggle.py 풀어둔폴더 --dry-run      # 어느 폴더를 코골이로 보는지 먼저 확인
3. python tools/import_kaggle.py 풀어둔폴더

폴더 이름으로 자동 판단:  1 / snoring / snore → 코골이,   0 / non / no / other → 아닌 소리
자동 판단이 틀리면:       --snore-dir 폴더이름 --other-dir 폴더이름
(꼭 파일 몇 개를 직접 들어 보고 확인할 것)

저장 위치: data/subjects/KG1..KG5/snore, noise  (파일 번호 순서로 5묶음)
  묶음 = 교차검증의 '참가자' 역할. 같은 출처(웹 소리)의 연속 파일이 학습·시험에 섞이는 것을 줄임
  클립마다 다른 사람·방이라 바닥소음은 클립별로 계산 (config.PER_CLIP_FLOOR_PREFIX)
"""
import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snore.config import SR  # noqa: E402
from tools.import_data import read_audio  # noqa: E402

SNORE_NAMES = {"1", "snoring", "snore", "snores", "코골이"}
OTHER_NAMES = {"0", "non-snoring", "non_snoring", "nonsnoring", "non", "no", "no_snoring", "other", "noise"}


def find_dirs(root):
    found = {}
    for dp, dirs, files in os.walk(root):
        name = os.path.basename(dp).lower()
        has_audio = any(f.lower().endswith((".wav", ".mp3", ".m4a")) for f in files)
        if not has_audio:
            continue
        if name in SNORE_NAMES:
            found.setdefault("snore", dp)
        elif name in OTHER_NAMES:
            found.setdefault("noise", dp)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--snore-dir")
    ap.add_argument("--other-dir")
    ap.add_argument("--subject", default="KG")
    ap.add_argument("--groups", type=int, default=5,
                    help="파일 번호 순서대로 N묶음(KG1..KGN)으로 나눔 → 교차검증이 '다른 묶음'으로 시험 (같은 출처끼리 섞이는 것 줄임). 1 = 묶지 않음")
    ap.add_argument("--limit", type=int, default=0, help="종류별 최대 개수 (0 = 전부)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dirs = find_dirs(a.folder)
    if a.snore_dir:
        dirs["snore"] = os.path.join(a.folder, a.snore_dir)
    if a.other_dir:
        dirs["noise"] = os.path.join(a.folder, a.other_dir)
    if set(dirs) != {"snore", "noise"}:
        raise SystemExit(f"폴더를 못 찾았어요: {dirs}\n--snore-dir / --other-dir 로 알려 주세요")

    for lab, d in dirs.items():
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".wav", ".mp3", ".m4a")))
        if a.limit:
            files = files[: a.limit]
        print(f"{'코골이' if lab == 'snore' else '아닌 소리'} ← {d}  ({len(files)}개)")
        if a.dry_run:
            continue
        import re
        num = lambda f: int(re.findall(r"\d+", f)[-1]) if re.findall(r"\d+", f) else 0
        files = sorted(files, key=num)
        lens = []
        for i, f in enumerate(files):
            g = a.subject if a.groups <= 1 else f"{a.subject}{i * a.groups // len(files) + 1}"
            out = os.path.join(a.data, "subjects", g, lab)
            os.makedirs(out, exist_ok=True)
            x, _ = read_audio(os.path.join(d, f))
            lens.append(len(x) / SR)
            wavfile.write(os.path.join(out, os.path.splitext(f)[0] + ".wav"), SR,
                          (np.clip(x, -1, 1) * 32767).astype(np.int16))
        print(f"  저장 {os.path.join(a.data, 'subjects', a.subject + '*', lab)}  ({len(files)}개, 평균 {np.mean(lens):.2f}초, {max(1, a.groups)}묶음)")
    if not a.dry_run:
        print("\n다음: python train.py   (결과표에 KG 참가자 = 공개 데이터 시험 결과)")


if __name__ == "__main__":
    main()
