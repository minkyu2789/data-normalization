# 5. 데이터 넣기 가이드 (처음부터 끝까지)

## 0. 준비 (한 번만)

```
pip install -r requirements.txt
pip install imageio-ffmpeg          # 휴대폰 m4a 파일을 읽을 때 필요
```

## 1. 가짜 데이터로 먼저 돌려 보기 (파일 넣을 필요 없음)

```
python train.py --demo
```

`data/` 폴더에 가짜 소리가 자동으로 생기고 학습까지 진행된다. 실제 데이터를 넣을 때는 `data/` 폴더를 지우고 시작한다.

## 2. 휴대폰 녹음 넣기 (폴더째)

1. 녹음 파일 이름에 **참가자 번호 + 소리 종류**를 적는다

```
녹음/
├─ P01_코골이_1.m4a
├─ P01_코골이_2.m4a
├─ P01_이불코골이.m4a
├─ P01_옷소리.m4a
├─ P01_뒤척임.m4a
├─ P01_말소리.m4a
├─ P01_조용함.m4a
└─ P02/ … (폴더 이름으로 참가자를 구분해도 됨)
```

2. 먼저 어떻게 분류되는지 확인 → 넣기 → 점검 → 학습

```
python tools/import_folder.py 녹음 --dry-run
python tools/import_folder.py 녹음
python tools/check_data.py
python train.py
```

| 파일 이름에 있으면 | 분류 |
|---|---|
| 코골 + 이불 (또는 덮) | 이불 덮고 코골이 |
| 코골 | 코골이 |
| 옷, 이불 | 옷·이불 소리 |
| 뒤척, 움직 | 뒤척임 |
| 말, 잠꼬대 | 말소리 |
| 조용, 무음 | 조용함 |

## 3. 여러 소리가 섞인 긴 녹음 (실제 밤 녹음)

실제 밤에는 6가지 소리가 섞여 나온다. 두 가지 용도가 있다.

**A. 학습에 쓰기**: 정답 표시가 필요하다

1. Audacity로 녹음을 연다
2. 코골이 구간을 드래그 → `Ctrl+B` → `snore` 입력 (옷 소리면 `cloth` …)
3. 파일 → 내보내기 → 라벨 내보내기 → 녹음과 **같은 이름 .txt** 로 저장
4. `python tools/import_folder.py 녹음` (같은 이름 .txt 가 있으면 긴 녹음으로 들어감)

라벨 이름: `snore` `snore_covered` `cloth` `movement` `speech` `quiet`

**B. 탐지·측정하기**: 정답 없이 바로

```
python detect.py --wav 밤녹음.wav --model rf
```

출력:

- 0.5초마다 무슨 소리인지 → `timeline_*_sounds.csv` (시작, 끝, 소리 종류)
- 코골이 측정 요약 → `timeline_*_summary.csv` (코골이 시간·비율, 시간당 구간 수, 크기)
- 타임라인 그림 (맨 아래 줄 = 소리 종류 색)

정답(.txt)을 같이 주면 정확도도 계산한다: `--labels 밤녹음.txt`

## 4. 우리 측정 장치 기록 넣기

```
python tools/import_session.py collections/20261112_0128 --subject P01
python tools/import_session.py collections --subject P01 --all
```

`audio.wav` → 코골이용, `emg.npy` → 이갈이용 (`data/emg/`), `labels.csv` → 정답

## 5. Kaggle 공개 코골이 데이터 (코골이 500 + 아닌 소리 500, 각 1초)

1. https://www.kaggle.com/ 가입·로그인
2. https://www.kaggle.com/datasets/tareqkhanemu/snoring 접속 → **Download** → zip 풀기
3. 폴더 확인 → 넣기

```
python tools/import_kaggle.py 풀어둔폴더 --dry-run
python tools/import_kaggle.py 풀어둔폴더
python train.py
```

- 받은 zip 구조: `Snoring Dataset/1` = 코골이 500개, `Snoring Dataset/0` = 아닌 소리 500개 (데이터셋 설명 파일로 확인). 1초, 44.1/48 kHz → 16 kHz로 자동 변환
- 파일 번호 순서로 5묶음(`KG1`~`KG5`)으로 나눠 넣음 → 교차검증이 '다른 묶음'으로 시험 (같은 출처끼리 섞이는 것 줄임)
- 이 데이터로 이미 학습한 모델: `models_kaggle/` (`--models-dir models_kaggle` 로 바로 사용), 결과: `example_results_kaggle/`
- 아닌 소리는 10종류 × 50개: 아기 울음, 시계 소리, 문 여닫기, 무음·진동 모터, 변기 물, 사이렌, 비·천둥, 전차, 사람 말소리, TV 뉴스 → 라벨 `noise`(기타 잡음)
- 다양한 사람의 코골이를 배우는 데 도움. 우리 녹음과 섞어서 쓰는 것이 목적

## 6. 이갈이 공개 데이터

**Kaggle에는 쓸 만한 이갈이 근전도 데이터가 없다.** 대안:

| 데이터 | 내용 | 한계 |
|---|---|---|
| PhysioNet CAP Sleep Database `brux1`, `brux2` | 이갈이 환자 2명의 밤 수면다원검사 (턱 밑 근전도 포함, 로그인 없이 공개) | 이갈이 구간 정답 표시 없음 (수면 단계만). 전극 위치가 우리와 다름 |

```
# https://physionet.org/content/capslpdb/1.0.0/ 에서 brux1.edf 받기 (약 220 MB)
python tools/import_edf.py brux1.edf --list
python tools/import_edf.py brux1.edf
python brux_detect.py --emg data/emg/CAP_brux1_emg.npy --warmup 0
```

→ 실제 사람 근전도에서 탐지기가 어떻게 동작하는지 "보는" 용도. 정확도는 우리 보정 기록(`brux_calibrate.py`)으로 확인한다.

## 7. 이갈이 측정 순서 (우리 장치)

1. 전극 부착 (관자놀이/교근 +, 귓불 −), 노트북 배터리 + USB 아이솔레이터
2. 측정 시작 → `python brux_calibrate.py --guide --subject P01` 안내대로 약 3분
3. `python brux_calibrate.py --emg 세션/emg.npy --marks results/calib_P01_marks.csv --subject P01`
   → 모든 동작에 ✓ 가 나오면 보정 통과
4. 밤 측정 → `python brux_detect.py --emg 세션/emg.npy --calib results/brux_calib_P01.json`
