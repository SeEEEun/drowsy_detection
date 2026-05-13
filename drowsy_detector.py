"""
drowsy_detector.py
운전자 상태 감지 시스템 - Car A (OpenCV 파트)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
역할: 카메라 → OpenCV 분석 → 상태 코드(Level 1/2/3) 출력
출력 채널 (택 1 또는 동시):
  - 시리얼 통신 (라즈베리파이로 전송)
  - MQTT publish (브로커로 직접 전송)
  - 콘솔 출력 (개발/디버깅용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
감지 알고리즘:
  1. EAR - 눈 감김 지속 시간  (+3점)
  2. MAR - 하품 감지          (+2점)
  3. NOD - 고개 끄덕임        (+3점)
  → 복합 점수 기반 Level 판정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import collections
import json

# ═══════════════════════════════════════════════════
#  통신 설정 (회의 결과에 따라 켜고 끄기)
# ═══════════════════════════════════════════════════
USE_SERIAL = False    # True면 시리얼 전송 활성화
USE_MQTT   = False    # True면 MQTT publish 활성화
USE_CONSOLE = True    # 콘솔 출력 (개발 중엔 켜두기)

# [시리얼] 라즈베리파이 연결
SERIAL_PORT = "/dev/ttyUSB0"   # Windows: "COM3", Mac: "/dev/tty.usbserial-XXXX"
SERIAL_BAUD = 9600

# [MQTT] 직접 publish 할 경우
MQTT_BROKER = "localhost"      # 준재가 알려준 브로커 주소
MQTT_PORT   = 1883
MQTT_TOPIC  = "vehicle/1/status"   # 1호차 상태 토픽

# [차량 정보]
VEHICLE_ID = 1                  # 1호차 / 2호차 / 3호차

# ═══════════════════════════════════════════════════
#  감지 설정값 (튜닝 가능)
# ═══════════════════════════════════════════════════

# [EAR] 눈 감김
EAR_THRESHOLD      = 0.20
EAR_CONSEC_SECONDS = 1.5

# [MAR] 하품
MAR_THRESHOLD      = 0.65
MAR_CONSEC_SECONDS = 1.5

# [NOD] 고개 끄덕임
NOD_Y_THRESHOLD    = 12
NOD_COUNT_TRIGGER  = 3
NOD_WINDOW_SECONDS = 6.0

# [SCORE] 복합 점수
SCORE_EAR          = 3
SCORE_MAR          = 2
SCORE_NOD          = 3
SCORE_DECAY        = 0.5

# [LEVEL] 상태 코드 임계값
LEVEL_2_THRESHOLD  = 4      # 점수 ≥ 4 → Level 2 (주의)
LEVEL_3_THRESHOLD  = 7      # 점수 ≥ 7 → Level 3 (전투력 상실)

# [전송 정책]
SEND_INTERVAL_SECONDS = 5.0   # heartbeat 주기 (정상 상태일 때 송신 간격)
LEVEL3_COOLDOWN       = 5.0   # Level 3 신호 전송 후 재전송 대기
CONSOLE_ON_CHANGE_ONLY = True  # 콘솔은 레벨 바뀔 때만 출력 (False면 매번 출력)

# ═══════════════════════════════════════════════════
#  통신 초기화
# ═══════════════════════════════════════════════════
ser = None
mqtt_client = None

if USE_SERIAL:
    import serial
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)  # 시리얼 안정화 대기
        print(f"✅ 시리얼 연결: {SERIAL_PORT} @ {SERIAL_BAUD}")
    except Exception as e:
        print(f"❌ 시리얼 연결 실패: {e}")
        ser = None

if USE_MQTT:
    import paho.mqtt.client as mqtt
    try:
        mqtt_client = mqtt.Client(client_id=f"vehicle_{VEHICLE_ID}_opencv")
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"✅ MQTT 연결: {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"❌ MQTT 연결 실패: {e}")
        mqtt_client = None

# ═══════════════════════════════════════════════════
#  Level 판정 함수
# ═══════════════════════════════════════════════════
def get_level(score):
    """위험 점수 → Level 1/2/3 변환"""
    if score >= LEVEL_3_THRESHOLD:
        return 3
    elif score >= LEVEL_2_THRESHOLD:
        return 2
    else:
        return 1

def level_to_korean(level):
    return {1: "정상", 2: "주의", 3: "전투력 상실"}.get(level, "Unknown")

# ═══════════════════════════════════════════════════
#  상태 코드 전송 (1KB 미만 페이로드)
# ═══════════════════════════════════════════════════
def send_status(level, score, reason="", level_changed=False):
    """모든 활성 채널로 상태 코드 송신"""
    payload = {
        "vid":   VEHICLE_ID,
        "lvl":   level,
        "score": round(score, 1),
        "reason": reason,
        "ts":    int(time.time())
    }
    msg = json.dumps(payload, ensure_ascii=False)

    # 시리얼 전송 (라즈베리파이로)
    if ser is not None:
        try:
            ser.write((msg + "\n").encode("utf-8"))
        except Exception as e:
            print(f"⚠️ 시리얼 송신 실패: {e}")

    # MQTT publish
    if mqtt_client is not None:
        try:
            mqtt_client.publish(MQTT_TOPIC, msg, qos=1)
        except Exception as e:
            print(f"⚠️ MQTT 송신 실패: {e}")

    # 콘솔 출력 (레벨 바뀔 때만, 또는 매번)
    if USE_CONSOLE:
        if (not CONSOLE_ON_CHANGE_ONLY) or level_changed:
            icon = {1: "🟢", 2: "🟡", 3: "🔴"}[level]
            print(f"{icon} [Level {level} - {level_to_korean(level)}] {msg}")

# ═══════════════════════════════════════════════════
#  MediaPipe 초기화
# ═══════════════════════════════════════════════════
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
NOSE_TIP  = 1

# ═══════════════════════════════════════════════════
#  계산 함수
# ═══════════════════════════════════════════════════
def calc_ear(landmarks, eye_idx, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def calc_mar(landmarks, w, h):
    pts = {
        'top':   (landmarks[13].x  * w, landmarks[13].y  * h),
        'bot':   (landmarks[14].x  * w, landmarks[14].y  * h),
        'left':  (landmarks[78].x  * w, landmarks[78].y  * h),
        'right': (landmarks[308].x * w, landmarks[308].y * h),
        'lt':    (landmarks[82].x  * w, landmarks[82].y  * h),
        'rb':    (landmarks[312].x * w, landmarks[312].y * h),
    }
    A = distance.euclidean(pts['top'],  pts['bot'])
    B = distance.euclidean(pts['lt'],   pts['rb'])
    C = distance.euclidean(pts['left'], pts['right'])
    return (A + B) / (2.0 * C)

# ═══════════════════════════════════════════════════
#  상태 변수
# ═══════════════════════════════════════════════════
eye_closed_start = None
ear_scored       = False

mouth_open_start = None
mar_scored       = False
yawn_count       = 0

nod_times        = collections.deque()
prev_nose_y      = None
nod_direction    = None
nod_scored       = False

danger_score      = 0.0
score_last_update = time.time()

current_level     = 1
last_sent_level   = 1
last_send_time    = 0.0
last_level3_time  = 0.0
last_reason       = ""

# ═══════════════════════════════════════════════════
#  카메라 초기화
# ═══════════════════════════════════════════════════
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)         # 프레임레이트 명시
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 버퍼 1개만 → 지연 최소화
time.sleep(1.0)
for _ in range(5):
    cap.read()

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"  Vehicle ID  : {VEHICLE_ID}")
print(f"  Level 2 기준: 점수 ≥ {LEVEL_2_THRESHOLD}")
print(f"  Level 3 기준: 점수 ≥ {LEVEL_3_THRESHOLD}")
print(f"  Serial      : {'ON' if ser else 'OFF'}")
print(f"  MQTT        : {'ON' if mqtt_client else 'OFF'}")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("✅ 감지 시작 | 종료: q키")

# ═══════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    now = time.time()

    # ── 점수 시간 감쇠 ──
    dt = now - score_last_update
    danger_score = max(0.0, danger_score - SCORE_DECAY * dt)
    score_last_update = now

    on_level3_cooldown = (now - last_level3_time) < LEVEL3_COOLDOWN

    # ── 게이지 색상 ──
    ratio_score = min(danger_score / LEVEL_3_THRESHOLD, 1.0)
    if ratio_score < 0.4:
        score_color = (0, 200, 0)
    elif ratio_score < 0.75:
        score_color = (0, 200, 255)
    else:
        score_color = (0, 0, 255)

    reasons_this_frame = []

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )

        # [1] EAR
        l_ear = calc_ear(lms, LEFT_EYE,  w, h)
        r_ear = calc_ear(lms, RIGHT_EYE, w, h)
        ear   = (l_ear + r_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = now
                ear_scored       = False
            closed_dur = now - eye_closed_start

            ratio_ear = min(closed_dur / EAR_CONSEC_SECONDS, 1.0)
            bar_w = int(w * ratio_ear)
            cv2.rectangle(frame, (0, h-20), (bar_w, h),
                          (0, int(255*(1-ratio_ear)), int(255*ratio_ear)), -1)
            cv2.putText(frame, f"Eye Closed: {closed_dur:.1f}s / {EAR_CONSEC_SECONDS}s",
                        (10, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if closed_dur >= EAR_CONSEC_SECONDS and not ear_scored:
                danger_score += SCORE_EAR
                ear_scored    = True
                reasons_this_frame.append("EAR")
                print(f"[EAR] 눈 감김 → +{SCORE_EAR}점 (총 {danger_score:.1f})")
        else:
            eye_closed_start = None
            ear_scored       = False

        # [2] MAR
        mar = calc_mar(lms, w, h)

        if mar > MAR_THRESHOLD:
            if mouth_open_start is None:
                mouth_open_start = now
                mar_scored       = False
            open_dur = now - mouth_open_start

            cv2.putText(frame, f"Yawn: {open_dur:.1f}s / {MAR_CONSEC_SECONDS}s",
                        (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            if open_dur >= MAR_CONSEC_SECONDS and not mar_scored:
                danger_score  += SCORE_MAR
                mar_scored     = True
                yawn_count    += 1
                reasons_this_frame.append("MAR")
                print(f"[MAR] 하품 (총 {yawn_count}회) → +{SCORE_MAR}점 (총 {danger_score:.1f})")
        else:
            mouth_open_start = None
            mar_scored       = False

        # [3] NOD
        nose_y = lms[NOSE_TIP].y * h
        if prev_nose_y is not None:
            diff    = nose_y - prev_nose_y
            new_dir = None
            if diff >  NOD_Y_THRESHOLD: new_dir = "down"
            if diff < -NOD_Y_THRESHOLD: new_dir = "up"
            if new_dir and new_dir != nod_direction:
                nod_direction = new_dir
                nod_times.append(now)

        while nod_times and (now - nod_times[0]) > NOD_WINDOW_SECONDS:
            nod_times.popleft()
        nod_count = len(nod_times) // 2
        prev_nose_y = nose_y

        if nod_count >= NOD_COUNT_TRIGGER and not nod_scored:
            danger_score += SCORE_NOD
            nod_scored    = True
            reasons_this_frame.append("NOD")
            print(f"[NOD] 끄덕임 {nod_count}회 → +{SCORE_NOD}점 (총 {danger_score:.1f})")
            nod_times.clear()
        if nod_count < NOD_COUNT_TRIGGER:
            nod_scored = False

        # ══════════════════════════════
        #  [4] Level 판정
        # ══════════════════════════════
        current_level = get_level(danger_score)

        if reasons_this_frame:
            last_reason = "+".join(reasons_this_frame)

        # 전송 정책:
        #  - Level 바뀌면 즉시 전송
        #  - Level 3 처음 진입 시: 즉시 전송 + cooldown 시작 + 점수 리셋
        #  - 그 외엔 SEND_INTERVAL_SECONDS마다 heartbeat
        should_send = False

        if current_level != last_sent_level:
            should_send = True
        elif (now - last_send_time) >= SEND_INTERVAL_SECONDS:
            should_send = True

        if current_level == 3 and on_level3_cooldown:
            should_send = False  # cooldown 중엔 재전송 안 함

        if should_send:
            level_changed = (current_level != last_sent_level)
            send_status(current_level, danger_score, last_reason, level_changed)
            last_sent_level = current_level
            last_send_time  = now

            if current_level == 3 and not on_level3_cooldown:
                last_level3_time = now
                danger_score     = 0.0  # 신호 보낸 뒤 리셋

        # ── HUD ──
        cv2.putText(frame, f"EAR: {ear:.3f}  (< {EAR_THRESHOLD})",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}  (> {MAR_THRESHOLD})",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"Nod: {nod_count} / {NOD_COUNT_TRIGGER}",
                    (10, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"Yawn Count: {yawn_count}",
                    (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)

        if on_level3_cooldown:
            remain = LEVEL3_COOLDOWN - (now - last_level3_time)
            cv2.putText(frame, f"Cooldown: {remain:.1f}s",
                        (10, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    else:
        current_level = 1  # 얼굴 미감지 시 일단 정상으로 (혹은 별도 Level 처리)
        cv2.putText(frame, "No Face Detected",
                    (w//2 - 110, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)

    # ── 게이지 ──
    gauge_w = int(w * min(danger_score / LEVEL_3_THRESHOLD, 1.0))
    cv2.rectangle(frame, (0, 42), (w, 58), (50,50,50), -1)
    cv2.rectangle(frame, (0, 42), (gauge_w, 58), score_color, -1)
    cv2.putText(frame, f"Score: {danger_score:.1f} / {LEVEL_3_THRESHOLD}",
                (w-220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

    # ── 상태바 ──
    level_color = {1: (0,200,0), 2: (0,165,255), 3: (0,0,255)}[current_level]
    level_name_en = {1: "NORMAL", 2: "CAUTION", 3: "DANGER"}[current_level]
    status_text = f"Level {current_level} - {level_name_en}"
    cv2.rectangle(frame, (0,0), (w, 42), (30,30,30), -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)
    cv2.putText(frame, f"Vehicle {VEHICLE_ID}",
                (w-110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ═══════════════════════════════════════════════════
#  종료 처리
# ═══════════════════════════════════════════════════
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

if ser is not None:
    ser.close()
if mqtt_client is not None:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

print("감지 종료")
