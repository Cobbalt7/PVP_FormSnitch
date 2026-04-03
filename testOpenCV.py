import time
import cv2
import platform
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import os

# Squat counting / feedback
squat_counter = 0
squat_stage = "UP"
feedback = "No pose"

prev_knee_angle = 180
prev_back_angle = 180


def test_indices():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: WORKING ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"Index {i}: OPENED BUT NO FRAME (Check format/MJPG)")
            cap.release()
        else:
            print(f"Index {i}: NOT AVAILABLE")


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style
        )

    return annotated_image

def draw_landmark_indices(frame, landmarks):
    h, w, _ = frame.shape
    important_ids = [11, 23, 25, 27]

    for idx in important_ids:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.putText(frame, str(idx), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return 0

    cos_angle = np.dot(ba, bc) / denominator
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def analyze_squat(landmarks, prev_knee_angle, prev_back_angle, squat_stage, squat_counter):
    shoulder = landmarks[11]
    hip = landmarks[23]
    knee = landmarks[25]
    ankle = landmarks[27]
    used_side = "LEFT"

    if (shoulder.visibility < 0.5 or
        hip.visibility < 0.5 or
        knee.visibility < 0.5 or
        ankle.visibility < 0.5):
        return prev_knee_angle, prev_back_angle, squat_stage, squat_counter, "Landmarks not visible"

    knee_angle_raw = calculate_angle(hip, knee, ankle)
    back_angle_raw = calculate_angle(shoulder, hip, knee)

    alpha = 0.7
    knee_angle = alpha * prev_knee_angle + (1 - alpha) * knee_angle_raw
    back_angle = alpha * prev_back_angle + (1 - alpha) * back_angle_raw

    if knee_angle < 100 and squat_stage == "UP":
        squat_stage = "DOWN"
    elif knee_angle > 160 and squat_stage == "DOWN":
        squat_stage = "UP"
        squat_counter += 1

    if knee_angle > 120:
        feedback = f"{used_side}: Too shallow"
    elif back_angle < 145:
        feedback = f"{used_side}: Lean less forward"
    else:
        feedback = f"{used_side}: Good squat"

    return knee_angle, back_angle, squat_stage, squat_counter, feedback

def draw_analysis_lines(frame, landmarks):
    h, w, _ = frame.shape

    shoulder = landmarks[11]
    hip = landmarks[23]
    knee = landmarks[25]
    ankle = landmarks[27]

    pts = []
    for lm in [shoulder, hip, knee, ankle]:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))

    shoulder_pt, hip_pt, knee_pt, ankle_pt = pts

    # Nugara / liemuo
    cv2.line(frame, shoulder_pt, hip_pt, (255, 0, 0), 3)

    # �launis
    cv2.line(frame, hip_pt, knee_pt, (0, 255, 255), 3)

    # Blauzda
    cv2.line(frame, knee_pt, ankle_pt, (255, 255, 0), 3)

    # Ta�kai
    for pt in pts:
        cv2.circle(frame, pt, 6, (0, 0, 255), -1)

class LandmarkerAndResult:
    def __init__(self):
        self.result = None
        self.landmarker_class = mp.tasks.vision.PoseLandmarker
        self.create_landmarker()

    def create_landmarker(self):
        def update_result(result: mp.tasks.vision.PoseLandmarkerResult,
                          output_image: mp.Image,
                          timestamp_ms: int):
            self.result = result

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=update_result
        )

        self.landmarker = self.landmarker_class.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(
            image=mp_image,
            timestamp_ms=int(time.time() * 1000)
        )

    def close(self):
        self.landmarker.close()


def main():
    global squat_counter, squat_stage, feedback
    global prev_knee_angle, prev_back_angle

    # test_indices()

    if platform.system() == 'Linux':
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    elif platform.system() == 'Windows':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera could not be opened")
        return

    pose_landmarker = LandmarkerAndResult()

    # Warm up camera
    for _ in range(5):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)
        pose_landmarker.detect_async(frame)

        if pose_landmarker.result is not None and len(pose_landmarker.result.pose_landmarks) > 0:
            landmarks = pose_landmarker.result.pose_landmarks[0]

            prev_knee_angle, prev_back_angle, squat_stage, squat_counter, feedback = analyze_squat(
                landmarks,
                prev_knee_angle,
                prev_back_angle,
                squat_stage,
                squat_counter
            )

            frame = draw_landmarks_on_image(frame, pose_landmarker.result)
            draw_landmark_indices(frame, landmarks)
            draw_analysis_lines(frame, landmarks)

            cv2.putText(frame, f"Reps: {squat_counter}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Knee angle: {int(prev_knee_angle)}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Back angle: {int(prev_back_angle)}",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, f"Stage: {squat_stage}",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Feedback: {feedback}",
                        (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No pose detected",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Squat Form Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    pose_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()