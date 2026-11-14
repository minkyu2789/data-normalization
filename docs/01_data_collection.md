# 1. 데이터 수집

## 1-1. 무엇을 모으나

| 센서 | 저장 형식 | 샘플링 | 단위 | 쓰는 곳 |
|---|---|---|---|---|
| 마이크 INMP441 | **WAV** (16-bit, 모노) | 16 kHz | -1~1 (dBFS) | 코골이 분류 |
| 턱/관자놀이 EMG (ADS1292R) | **CSV** `t_s, ch1_uV` | 1000 SPS | µV | 이갈이 탐지 |
| 정답 표시 | **CSV** `t_s,label` 또는 Audacity `.txt` | 사람이 누른 순간 | – | 학습·평가 정답 |
| 트리거 판단 기록 | **CSV** (`*_decisions.csv`) | 판단할 때마다 | – | 자극 요청 근거 |

- 마이크 소리를 **크기(dB)만 CSV로 저장하면 코골이 분류에 쓸 수 없음** → 원본 WAV 필수
- 4시간 기준 용량: WAV 약 460 MB, EMG CSV 약 1~2 GB

## 1-2. 수집 절차 (1명, 약 40분 + 밤 측정)

**A. 낮 녹음 (라벨이 확실한 짧은 녹음)** – 휴대폰 또는 장치 마이크, 머리에서 30~50 cm

| 라벨 | 폴더 이름 | 길이 | 방법 |
|---|---|---|---|
| 코골이 | `snore` | 2분 × 2회 | 누워서 코골이 흉내 (입 다물고 / 벌리고) |
| 이불 덮고 코골이 | `snore_covered` | 2분 | 이불을 머리까지 덮고 |
| 옷·이불 소리 | `cloth` | 1분 | 이불 비비기, 옷 스치기 |
| 뒤척임 | `movement` | 1분 | 돌아눕기, 베개 소리 |
| 말소리 | `speech` | 1분 | 잠꼬대처럼 중얼거림 |
| 조용함 | `quiet` | 1분 | 아무것도 안 함 (바닥소음 기준) |

**B. 이갈이 보정 (EMG)** – 전극 붙이고 깨어 있는 상태

1. 휴식 60초 (기준선)
2. **최대로 꽉 물기 3초 × 3회** (사이 5초 휴식) → `calib_emg.csv` → MVC(100%) 기준
3. 이 갈기 흉내 10초, 침 삼키기 5회, 하품·말하기 (오탐 확인용)

**C. 밤 측정** – `collect.py`(동기 수집기) 또는 recorder로 WAV + EMG 동시 저장, 일어나서 기억나는 사건만 메모

## 1-3. 정답(라벨) 붙이기

- **짧은 녹음**: 폴더 이름이 곧 라벨 (`data/subjects/P01/snore/001.wav`)
- **긴 녹음**: Audacity에서 구간 선택 → `Ctrl+B` → 라벨 입력 → 파일 > 내보내기 > 라벨 → 같은 이름 `.txt`
- **측정 중 키 입력**: `t_s,label` 형식, `snore_start` / `snore_end` 또는 같은 라벨 두 번 → 자동으로 구간 변환

## 1-4. 폴더에 넣기 (`tools/import_data.py`)

```
python tools/import_data.py 휴대폰.wav --subject P01 --label snore
python tools/import_data.py night.wav  --subject P02 --marks marks.csv
python tools/import_data.py audio.bin  --raw int16 --sr 16000 --subject P03 --marks marks.csv
python tools/check_data.py                       # 품질 점검 (클리핑, 부족한 라벨, 라벨 범위)
```

- 어떤 샘플레이트/스테레오든 16 kHz 모노로 변환, 직류 제거
- 라벨 이름 통일: `코골이 → snore`, `옷 → cloth` 등
- **참가자 번호(P01…)는 사람마다 다르게** → 평가를 사람 단위로 나누기 때문

## 1-5. 개인정보 · 동의

- 녹음에 말소리가 들어감 → 참가자 동의서, 파일은 팀 공유 드라이브에만, 발표에는 그림만
- 사람 대상 시험(본인 포함)은 학교 IRB 절차 확인 후 진행
