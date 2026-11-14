"""공통 설정"""

SR = 16000                # 샘플링 (Hz)
WIN_S = 1.0               # 분석 창 길이 (초)
HOP_S = 0.5               # 창 이동 (초) → 0.5초마다 판단
BAND = (50, 4000)         # 대역통과 필터 (Hz) - 코골이·잡음이 대부분 이 안에 있음

N_FFT = 512               # 32ms
HOP_LEN = 160             # 10ms
N_MELS = 64
N_MFCC = 20

# 라벨: 폴더 이름이나 Audacity 라벨에 쓰는 이름
SNORE_LABELS = {"snore", "snore_covered", "코골이", "코골이_이불"}
NOISE_LABELS = {"cloth", "movement", "speech", "quiet", "noise", "옷", "뒤척임", "말소리", "조용함"}
LABEL_KO = {
    "snore": "코골이", "snore_covered": "코골이(이불)", "cloth": "옷·이불 소리",
    "movement": "뒤척임", "speech": "말소리", "quiet": "조용함", "noise": "기타 잡음(공개데이터)",
}

# 공개 데이터셋처럼 짧은 클립이 여러 사람·방에서 온 경우: 바닥소음을 클립마다 따로 계산
PER_CLIP_FLOOR_PREFIX = ("KG",)

# 후처리 / 트리거
PROB_THR = 0.5            # 코골이 확률 기준
SMOOTH = 3                # 창 3개(1초) 중앙값으로 흔들림 제거
EPISODE_MIN = 3           # 코골이 소리가 3번 이상 반복돼야 '코골이 확정'
INTERVAL = (1.5, 10.0)    # 반복 간격 (초) = 호흡 주기
TRIGGER_GAP_S = 60        # 트리거 최소 간격 (초)
TRIGGER_MAX_H = 20        # 시간당 최대 트리거 수


def is_snore(label: str) -> bool:
    return label.strip().lower() in SNORE_LABELS
