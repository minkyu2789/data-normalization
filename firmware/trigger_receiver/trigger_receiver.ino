// 자극 요청 수신기 + 진동 피드백 (ESP32-S3) — PC의 live.py / trigger 와 짝
//
// PC → TRG,<번호>,<SNORE|BRUX>,<확신도>,<단계 1~3>*<CRC8>   HB,<번호>*<CRC8>   STOP*<CRC8>
// ESP → ACK,<번호>  /  NAK,<번호>,<이유>
//
// 출력: 코인형 진동 모터 (단계별 패턴) + 표시 LED. 전기 자극은 하지 않음.
//
// 배선 (진동 모터는 GPIO로 바로 돌리지 말 것: 전류가 커서 핀이 망가짐)
//   MOTOR_PIN ─ 1kΩ ─ NPN 트랜지스터(2N2222/S8050) 베이스   또는 로직레벨 N-MOSFET 게이트
//   모터 + ─ 3.3V(또는 5V)      모터 − ─ 트랜지스터 컬렉터(드레인),  이미터(소스) ─ GND
//   모터 양단에 다이오드(1N4148, 줄무늬가 + 쪽) 역방향으로 병렬
//   (드라이버 모듈 — 예: 진동 모터 모듈, DRV2605 — 을 써도 됨)
//
// 장치 쪽 안전 규칙 (PC 설정과 별개로 여기서도 검사)
//   - 심장박동(HB)이 3초 넘게 없으면 요청 거절 + 모터 정지
//   - 최소 간격 60초, 시간당 최대 20회
//   - 단계는 1~3만, 패턴과 최대 세기는 이 파일에 고정
//   - STOP 받으면 즉시 모터 정지, 이후 모든 요청 거절 (재부팅해야 풀림)
//
// 기존 펌웨어(cogoaltis_fw2)에 넣을 때: setup()의 내용 + loop()에서 pollTriggerSerial(); runPattern(); 호출
// (센서 패킷과 같은 시리얼을 쓰면 줄 구분이 섞이므로 Serial1 등 별도 포트 권장)
// ※ 컴파일 확인 전 코드. Arduino IDE, 보드 ESP32S3 Dev Module, ESP32 Arduino core 2.0 이상

const int LED_PIN = 2;
const int MOTOR_PIN = 4;                  // 사용 안 하는 핀으로 바꿔도 됨
const uint32_t HB_TIMEOUT_MS = 3000;
const uint32_t MIN_GAP_MS = 60000;
const int MAX_PER_HOUR = 20;

// 단계별 패턴: {켜는 시간 ms, 끄는 시간 ms, 반복 횟수, 세기 0~255}
struct Pattern { uint16_t onMs, offMs; uint8_t reps, duty; };
const Pattern PATTERNS[4] = {
  {0, 0, 0, 0},
  {200, 300, 2, 140},   // 1단계: 짧고 약하게
  {300, 300, 3, 200},   // 2단계
  {400, 300, 5, 255},   // 3단계: 가장 세게 (모터 정격 이내)
};

uint32_t lastHb = 0, lastTrig = 0;
uint32_t trigTimes[MAX_PER_HOUR];
int trigCount = 0;
bool stopped = false;
String buf;

// 진행 중인 패턴 (loop를 막지 않도록 시간으로 진행)
int curLevel = 0, curStep = 0;
uint32_t stepStart = 0;

uint8_t crc8(const char* s, int n) {
  uint8_t c = 0;
  for (int i = 0; i < n; i++) {
    c ^= (uint8_t)s[i];
    for (int k = 0; k < 8; k++) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
  }
  return c;
}

int recentCount(uint32_t now) {
  int n = 0;
  for (int i = 0; i < trigCount && i < MAX_PER_HOUR; i++)
    if (now - trigTimes[i] < 3600000UL) n++;
  return n;
}

void motorOff() {
  analogWrite(MOTOR_PIN, 0);
  digitalWrite(LED_PIN, LOW);
  curLevel = 0;
}

void startPattern(int level) {
  curLevel = level;
  curStep = 0;
  stepStart = millis();
}

void runPattern() {
  if (curLevel == 0) return;
  if (stopped || millis() - lastHb > HB_TIMEOUT_MS) { motorOff(); return; }
  const Pattern& p = PATTERNS[curLevel];
  uint32_t el = millis() - stepStart;
  int rep = curStep / 2;
  bool on = (curStep % 2 == 0);
  if (rep >= p.reps) { motorOff(); return; }
  analogWrite(MOTOR_PIN, on ? p.duty : 0);
  digitalWrite(LED_PIN, on ? HIGH : LOW);
  if (el >= (on ? p.onMs : p.offMs)) { curStep++; stepStart = millis(); }
}

void handle(const String& line) {
  int star = line.lastIndexOf('*');
  if (star < 0) return;
  String body = line.substring(0, star);
  uint8_t want = (uint8_t)strtol(line.substring(star + 1).c_str(), nullptr, 16);
  if (crc8(body.c_str(), body.length()) != want) return;

  uint32_t now = millis();
  if (body.startsWith("HB,")) { lastHb = now; return; }
  if (body == "STOP") { stopped = true; motorOff(); Serial.println("ACK,STOP"); return; }
  if (!body.startsWith("TRG,")) return;

  // TRG,<번호>,<원인>,<확신도>,<단계>
  int a = body.indexOf(',', 4), b = body.indexOf(',', a + 1), c = body.indexOf(',', b + 1);
  String n = body.substring(4, a);
  int level = (c > 0) ? body.substring(c + 1).toInt() : 1;
  level = constrain(level, 1, 3);

  const char* why = nullptr;
  if (stopped) why = "stopped";
  else if (now - lastHb > HB_TIMEOUT_MS) why = "no_heartbeat";
  else if (lastTrig && now - lastTrig < MIN_GAP_MS) why = "gap";
  else if (recentCount(now) >= MAX_PER_HOUR) why = "hour_limit";
  else if (curLevel != 0) why = "busy";
  if (why) { Serial.printf("NAK,%s,%s\n", n.c_str(), why); return; }

  lastTrig = now;
  trigTimes[trigCount % MAX_PER_HOUR] = now;
  trigCount++;
  Serial.printf("ACK,%s\n", n.c_str());
  startPattern(level);
}

void pollTriggerSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { handle(buf); buf = ""; }
    else if (buf.length() < 64) buf += ch;
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  motorOff();
  Serial.begin(115200);
}

void loop() {
  pollTriggerSerial();
  runPattern();
}
