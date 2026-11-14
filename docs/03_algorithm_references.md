# 3. 분류 알고리즘과 참고 문헌

## 3-1. 코골이 분류

| 모델 | 입력 | 판단 방식 | 장점 / 단점 |
|---|---|---|---|
| 규칙 기반 | rel_db, 저·중음 비율, 평탄도 | 세 기준을 모두 넘으면 코골이 (기준값은 학습 데이터에서 F1 최대로 탐색) | 설명 쉬움, ESP32에서도 가능 / 이불·잡음에 약함 |
| 랜덤포레스트 | 특징 49개 (MFCC 20×평균·표준편차, 대역 비율, 중심, 롤오프, 평탄도, 영교차율, rel_db, peak_db) | 나무 300개 투표 | 적은 데이터에 강함, 중요 특징 확인 가능 |
| CNN | 로그 멜 64×101 + rel_db, peak_db | 합성곱 3단 → 확률 | 이불 덮은 소리처럼 모양이 바뀐 경우에 강함 / 데이터 많이 필요 |

**후처리 (공통)**: 확률 → 1초 중앙값 평활 → 이어진 창 = 소리 1개 → **1.5–10초 간격으로 3번 이상 반복 = 코골이 확정**.
코골이는 숨마다 반복된다는 성질을 이용해 한 번짜리 소리(옷, 기침)를 거름.

**평가**: 참가자 단위 교차검증(한 명을 빼고 학습 → 그 사람으로 시험), 창 단위 F1 + 구간 단위 탐지/오탐.

## 3-2. 이갈이 분류 (규칙 기반)

- 버스트: 10 %MVC 이상 0.25초 이상
- 위상성: 0.25–2초 버스트 3개 이상 / 긴장성: 2초 넘는 버스트 / 혼합
- 3초 넘게 쉬면 다른 에피소드, 한 번짜리(삼키기)는 제외
- 지표: 시간당 에피소드 수

## 3-3. 참고 문헌

**코골이 소리 탐지**

1. Dafna E, Tarasiuk A, Zigel Y. *Automatic detection of whole night snoring events using non-contact microphone.* PLoS ONE 8(12): e84139, 2013. doi:10.1371/journal.pone.0084139
   → 비접촉 마이크로 밤새 코골이 탐지. 우리 구조(대역 특징 + 분류기 + 구간 판단)의 근거
2. Duckitt WD, Tuomi SK, Niesler TR. *Automatic detection, segmentation and assessment of snoring from ambient acoustic data.* Physiological Measurement 27(10), 2006. (https://dsp.sun.ac.za/~trn/reports/duckitt+tuomi+niesler_pmeas06.pdf)
   → 주변 소리에서 코골이/비코골이 분류, MFCC 사용
3. Nakano H, Hirayama K, Sadamitsu Y, et al. *Monitoring sound to quantify snoring and sleep apnea severity using a smartphone: proof of concept.* J Clin Sleep Med 10(1):73–78, 2014. (PMC3869073)
   → 스마트폰 마이크만으로 코골이 정량화 → 휴대폰 녹음 수집의 근거
4. Khan T. *A deep learning model for snoring detection and vibration notification using a smart wearable gadget.* Electronics 8(9):987, 2019. doi:10.3390/electronics8090987
   → **코골이 CNN 탐지 → 웨어러블 진동 알림**. 우리 "탐지 → 트리거" 흐름과 가장 가까운 사례
5. Schuller B, et al. *The INTERSPEECH 2017 Computational Paralinguistics Challenge: Addressee, Cold & Snoring.* Interspeech 2017. (Munich-Passau Snore Sound Corpus)
   → 공개 코골이 데이터셋, 데이터 늘릴 때 후보
6. Kong Q, et al. *PANNs: Large-scale pretrained audio neural networks for audio pattern recognition.* IEEE/ACM TASLP 28, 2020.
   → 로그 멜 + CNN 구조, 데이터가 늘면 사전학습 모델로 확장 가능

**이갈이 · 근전도**

7. Lavigne GJ, Rompré PH, Montplaisir JY. *Sleep bruxism: validity of clinical research diagnostic criteria in a controlled polysomnographic study.* J Dent Res 75(1):546–552, 1996. (PubMed 8655758)
   → 위상성/긴장성 에피소드 기준의 출처
8. Lobbezoo F, Ahlberg J, Raphael KG, et al. *International consensus on the assessment of bruxism: Report of a work in progress.* J Oral Rehabil 45(11):837–844, 2018. (PMC6287494)
   → 이갈이 평가 등급(자가보고 / 임상 / 기기 측정) 정리
9. Hermens HJ, et al. *Development of recommendations for SEMG sensors and sensor placement procedures (SENIAM).* J Electromyogr Kinesiol 10(5):361–374, 2000.
   → 표면 근전도 필터 대역·전극 배치 권고

**탐지 후 피드백(자극) 연구**

10. Raphael KG, Janal MN, Sirois DA, Svensson P, et al. *Effect of contingent electrical stimulation on masticatory muscle activity and pain in patients with a myofascial temporomandibular disorder and sleep bruxism.* J Orofac Pain 27(1):21–31, 2013. (PubMed 23424717)
    → 근전도로 이갈이를 감지할 때만 자극(contingent). **승인된 기기·임상 연구 환경**에서 수행
11. Jadidi F, et al. 수면 중 contingent electrical stimulation 파일럿 무작위 대조 연구. Acta Odontol Scand, 2013. doi:10.3109/00016357.2012.741702
12. Eijsvogel MM, et al. *Sleep position trainer versus tennis ball technique in positional obstructive sleep apnea syndrome.* J Clin Sleep Med 11(2):139–147, 2015. (PMC4298771)
    → 전기가 아닌 **진동 피드백**으로 수면 자세를 바꾸는 사례 → 코골이 쪽 대안

> 발표할 때: "우리 점수는 합성 데이터 기준이며, 실제 녹음으로 다시 측정할 예정"이라고 꼭 밝히기.
