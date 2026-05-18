"""
drowsy_detector.py
졸음/음주운전 감지 시스템 - Car A
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
감지 알고리즘:
  1. EAR - 눈 감김 지속 시간  (+3점)
  2. MAR - 하품 감지          (+2점)
  3. NOD - 고개 끄덕임        (+3점)
  🏆 복합 점수 7점 이상 → DANGER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import collections

# ═══════════════════════════════════════════════════
#  설정값 (튜닝 가능)
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
SCORE_EAR      = 3
SCORE_MAR      = 2
SCORE_NOD      = 3
SCORE_DANGER   = 7
SCORE_DECAY    = 0.5

# [LEVEL]
LEVEL_2_THRESHOLD = 4
LEVEL_3_THRESHOLD = 7

# [기타]
COOLDOWN_SECONDS = 5.0

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

def get_level(score):
    if score >= LEVEL_3_THRESHOLD:
        return 3
    elif score >= LEVEL_2_THRESHOLD:
        return 2
    else:
        return 1

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
last_danger_time  = 0.0

# ═══════════════════════════════════════════════════
#  카메라 초기화
# ═══════════════════════════════════════════════════
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1.0)
for _ in range(5):
    cap.read()

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  EAR  눈 감김  : +3점")
print("  MAR  하품     : +2점")
print("  NOD  끄덕임   : +3점")
print("  DANGER 기준   :  7점 이상")
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

    # 점수 감쇠
    dt            = now - score_last_update
    danger_score  = max(0.0, danger_score - SCORE_DECAY * dt)
    score_last_update = now

    on_cooldown = (now - last_danger_time) < COOLDOWN_SECONDS

    # 게이지 색상
    ratio_score = min(danger_score / LEVEL_3_THRESHOLD, 1.0)
    if ratio_score < 0.4:
        score_color = (0, 200, 0)
    elif ratio_score < 0.75:
        score_color = (0, 200, 255)
    else:
        score_color = (0, 0, 255)

    if results.multi_face_landmarks:
        lms = results.multi_face_landmarks[0].landmark

        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )

        # ── [1] EAR ──
        l_ear = calc_ear(lms, LEFT_EYE,  w, h)
        r_ear = calc_ear(lms, RIGHT_EYE, w, h)
        ear   = (l_ear + r_ear) / 2.0

        if ear < EAR_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = now
                ear_scored       = False
            closed_dur = now - eye_closed_start
            ratio_ear  = min(closed_dur / EAR_CONSEC_SECONDS, 1.0)
            bar_w      = int(w * ratio_ear)
            cv2.rectangle(frame, (0, h-20), (bar_w, h),
                          (0, int(255*(1-ratio_ear)), int(255*ratio_ear)), -1)
            cv2.putText(frame, f"Eye Closed: {closed_dur:.1f}s / {EAR_CONSEC_SECONDS}s",
                        (10, h-28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if closed_dur >= EAR_CONSEC_SECONDS and not ear_scored:
                danger_score += SCORE_EAR
                ear_scored    = True
                print(f"[EAR] 눈 감김 → +{SCORE_EAR}점 (총 {danger_score:.1f})")
        else:
            eye_closed_start = None
            ear_scored       = False

        # ── [2] MAR ──
        mar = calc_mar(lms, w, h)

        if mar > MAR_THRESHOLD:
            if mouth_open_start is None:
                mouth_open_start = now
                mar_scored       = False
            open_dur = now - mouth_open_start
            cv2.putText(frame, f"Yawn: {open_dur:.1f}s / {MAR_CONSEC_SECONDS}s",
                        (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            if open_dur >= MAR_CONSEC_SECONDS and not mar_scored:
                danger_score  += SCORE_MAR
                mar_scored     = True
                yawn_count    += 1
                print(f"[MAR] 하품 (총 {yawn_count}회) → +{SCORE_MAR}점 (총 {danger_score:.1f})")
        else:
            mouth_open_start = None
            mar_scored       = False

        # ── [3] NOD ──
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
        nod_count   = len(nod_times) // 2
        prev_nose_y = nose_y

        if nod_count >= NOD_COUNT_TRIGGER and not nod_scored:
            danger_score += SCORE_NOD
            nod_scored    = True
            print(f"[NOD] 끄덕임 {nod_count}회 → +{SCORE_NOD}점 (총 {danger_score:.1f})")
            nod_times.clear()
        if nod_count < NOD_COUNT_TRIGGER:
            nod_scored = False

        # ── [4] Level 판정 ──
        level = get_level(danger_score)

        if level == 3 and not on_cooldown:
            print(f"🚨 [DANGER] 위험점수: {danger_score:.1f}")
            last_danger_time = now
            danger_score     = 0.0

        # ── HUD ──
        cv2.putText(frame, f"EAR: {ear:.3f}  (< {EAR_THRESHOLD})",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"MAR: {mar:.3f}  (> {MAR_THRESHOLD})",
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"Nod: {nod_count} / {NOD_COUNT_TRIGGER}",
                    (10, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        cv2.putText(frame, f"Yawn: {yawn_count}회",
                    (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,0), 1)
        if on_cooldown:
            remain = COOLDOWN_SECONDS - (now - last_danger_time)
            cv2.putText(frame, f"Cooldown: {remain:.1f}s",
                        (10, 157), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    else:
        level = 1
        cv2.putText(frame, "No Face Detected",
                    (w//2-110, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)

    # ── 게이지 ──
    gauge_w = int(w * min(danger_score / LEVEL_3_THRESHOLD, 1.0))
    cv2.rectangle(frame, (0, 42), (w, 58), (50,50,50), -1)
    cv2.rectangle(frame, (0, 42), (gauge_w, 58), score_color, -1)
    cv2.putText(frame, f"Score: {danger_score:.1f} / {LEVEL_3_THRESHOLD}",
                (w-200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

    # ── 상태바 ──
    level_color = {1: (0,200,0), 2: (0,165,255), 3: (0,0,255)}[level]
    level_text  = {1: "Level 1 - NORMAL", 2: "Level 2 - CAUTION", 3: "Level 3 - DANGER"}[level]
    cv2.rectangle(frame, (0,0), (w, 42), (30,30,30), -1)
    cv2.putText(frame, level_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)
    cv2.putText(frame, "Car A - Driver Monitor",
                (w-230, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("감지 종료")
