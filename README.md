# 코골티스 감지·자극 소프트웨어

마이크 코골이 + 턱 근전도 이갈이 → 탐지 → 안전 게이트 → 단계별 진동 자극 → 효과 기록

```
소리(wav) → 전처리·정규화 → 분류(규칙/랜덤포레스트/CNN) → 3회 반복 = 코골이 확정 ┐
근전도    → 필터·%MVC/배경 기준 → 위상성·긴장성 = 이갈이 확정 ──────────────┴→ 안전 게이트 → 자극 1~3단계 → ESP32 진동
```

## 실제 데이터 결과 (Kaggle 공개 코골이 1,000개)

| 모델 | 정확도 | F1 | 비고 |
|---|---|---|---|
| 랜덤포레스트 | 0.883 | 0.899 | 개인 적응 시 F1 0.943 |
| CNN | 0.820 | 0.854 | 재현율 0.966 |
| 규칙 기반 | 0.781 | 0.784 | |

- 5묶음 교차검증 (학습에 없던 묶음으로 시험). 결과: `example_results_kaggle/`, 학습된 모델: `models_kaggle/`
- 코골이는 95% 찾음. 잡음 오인 18% — 특정 잡음 종류(파일 0_100–199, 0_450–499)에서 집중

## 문서 (docs/)

| 문서 | 내용 |
|---|---|
| `01_데이터수집.md` | 무엇을, 어떤 형식으로, 어떤 절차로 모으나 |
| `02_전처리_정규화.md` | 소리·근전도 전처리, 상대 크기, RMS·MVC 정규화, 코골이 판단 기준 |
| `03_분류알고리즘_레퍼런스.md` | 모델 3종, 이갈이 기준, 참고 문헌 |
| `04_자극트리거_소프트웨어.md` | 안전 게이트, 단계별 자극 알고리즘, PC↔ESP32 규약, 진동 모터 배선 |
| `05_데이터_넣기_가이드.md` | **휴대폰 녹음 / 섞인 밤 녹음 / 장치 기록 / Kaggle / PhysioNet 넣는 법** |

## 전체 명령

```
pip install -r requirements.txt
python train.py --demo                                     # 가짜 데이터로 전체 확인 (data/ 자동 생성)

# 데이터 넣기
python tools/import_folder.py 녹음폴더                      # 휴대폰 녹음 폴더째 (파일 이름으로 자동 분류)
python tools/import_session.py collections/세션 --subject P01  # 우리 장치 기록
python tools/import_kaggle.py kaggle폴더                    # Kaggle 공개 코골이
python tools/import_edf.py brux1.edf                        # PhysioNet 이갈이 환자 근전도
python tools/check_data.py                                  # 점검

# 코골이
python train.py                                             # 학습·평가·그림 (소리 6종 분류 포함)
python detect.py --wav 밤녹음.wav --model rf                 # 탐지 + 측정 요약 + 소리 종류 구간

# 이갈이
python brux_calibrate.py --guide --subject P01              # 보정 안내 (측정 시작 직후)
python brux_calibrate.py --emg 세션/emg.npy --marks results/calib_P01_marks.csv --subject P01
python brux_detect.py --emg 세션/emg.npy --calib results/brux_calib_P01.json

# 실시간 + 자극
python live.py --wav 밤녹음.wav --emg 세션/emg.npy --speed 0 --warmup 0   # 흐름 확인 (기록만)
python live.py --mic --port COM5 --arm                                   # 실제 진동
```

## 데이터 폴더 구조

```
data/
├─ subjects/                  # A. 소리 종류별 파일
│  └─ P01/
│     ├─ snore/*.wav          #    코골이
│     ├─ snore_covered/*.wav  #    이불 덮고 코골이
│     ├─ cloth/*.wav          #    옷·이불 비비는 소리
│     ├─ movement/*.wav       #    뒤척임
│     ├─ speech/*.wav         #    말소리
│     └─ quiet/*.wav          #    조용함 (바닥소음 기준으로도 사용)
├─ recordings/                # B. 길게 녹음 + Audacity 라벨 (학습용)
│  ├─ P02_night1.wav
│  └─ P02_night1.txt          #    시작<TAB>끝<TAB>라벨
└─ test/                      # 학습에 쓰지 않는 시험용 녹음
```

- 파일 이름 앞부분 `P01` = 참가자. 평가를 참가자 단위로 나누기 때문에 꼭 구분
- 아무 샘플레이트·모노/스테레오 wav 가능 (자동으로 16kHz 모노 변환)
- 휴대폰 녹음 → Audacity에서 구간 선택 후 Ctrl+B 로 라벨 → 파일 > 내보내기 > 라벨 내보내기

## 처리 순서

| 단계 | 내용 | 파일 |
|---|---|---|
| 1 | wav 불러오기, 16kHz 모노 변환 | `snore/dataio.py` |
| 2 | 대역통과 50–4000Hz, 바닥소음 추정, 1초 창(0.5초 이동), 상대 크기 계산, 창별 크기 정규화 | `snore/preprocess.py` |
| 3 | 라벨 정제: 코골이 녹음 안의 숨 사이 조용한 창(바닥소음 +6dB 미만)은 학습에서 제외 | `snore/pipeline.py` |
| 4 | 증강: 이불 효과, 배경 잡음, 크기 변화 | `snore/augment.py` |
| 5 | 특징: MFCC, 대역 에너지 비율, 스펙트럼 중심·평탄도, 영교차율, 상대 크기 (49개) / 로그 멜 스펙트로그램 | `snore/features.py` |
| 6 | 분류: 규칙 기반, 랜덤포레스트, CNN | `snore/models.py` |
| 7 | 평가: 참가자 단위 교차검증, 개인 적응 비교, 소리 종류별 판단 비율 | `snore/evaluate.py` |
| 8 | 후처리: 1초 중앙값 평활 → 소리 묶기 → 1.5–10초 간격 3회 반복 = 코골이 확정 → 트리거 (최소 60초 간격, 시간당 20회) | `snore/postprocess.py` |

## 결과물 (`results/`)

| 파일 | 내용 |
|---|---|
| `dataset_summary.csv` | 참가자 × 소리 종류별 녹음 시간 |
| `cv_results.csv` | 참가자별·모델별 정확도, 정밀도, 재현율, F1 |
| `results.md` | 자동 요약 리포트 |
| `fig1_patterns.png` | 소리 종류별 파형·스펙트로그램 |
| `fig2_features.png` | 특징 분포 (코골이 vs 잡음) |
| `fig3_confusion.png` | 혼동행렬 |
| `fig4_models.png` | 모델 비교 |
| `fig5_by_sound.png` | 소리 종류별 '코골이' 판단 비율 |
| `fig6_importance.png` | 랜덤포레스트 특징 중요도 |
| `fig7_sound_types.png` | 소리 6종 분류 혼동행렬 |
| `timeline_*.png`, `*_triggers.csv` | detect.py 탐지 타임라인(맨 아래 = 소리 종류), 트리거 기록 |
| `timeline_*_summary.csv`, `*_sounds.csv` | 코골이 측정 요약, 소리 종류 구간 |
| `brux_*.png`, `_summary.csv`, `_episodes.csv` | 이갈이 그림, 지표, 에피소드 |
| `live_*_decisions.csv`, `_stim.csv` | 실시간 판단 기록, 자극 단계·효과 |

## 주의
- `--demo` 데이터는 코드 확인용 합성 소리라 점수가 실제보다 높게 나옴. 성능은 실제 녹음으로 다시 측정할 것
- 자극 출력은 진동 모터 기준. 기본은 기록만, `live.py --port COMx --arm` 일 때만 ESP32로 전송
- 자작 전기 자극 회로(변압기 개조 등)는 지원하지 않음. 전기 자극은 인증 기기 + 안전 검토 후 (`docs/04`)
- `firmware/trigger_receiver.ino` 는 컴파일 확인 전 코드
