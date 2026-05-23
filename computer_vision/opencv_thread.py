import threading
import cv2
import queue
import mediapipe as mp
from cv2.typing import MatLike
import computer_vision.angles_and_evaluation as angev
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- THREAD 2: MACHINE LEARNING WORKER (The Heavy Lifter) ---
class OpenCVThread(threading.Thread):
    def __init__(self, raw_queue, processed_queue, running_event):
        super().__init__()
        self.raw_queue = raw_queue
        self.processed_queue = processed_queue
        self.running_event = running_event
        self.daemon=True
        self.model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.squat_tracker = angev.SquatTracker()

    def run(self):
        while True:
            self.running_event.wait()
            try:
                # 1. Grab the RAW frame (blocks until a frame is available)
                frame = self.raw_queue.get(timeout=1.0)
                
                # --------------------------------------------------------
                # RUN YOUR MACHINE LEARNING INFERENCE HERE
                # --------------------------------------------------------
                ml_result = self._infer_pose(frame)
                if ml_result is not None:
                    frame = self._draw_landmarks(frame, ml_result)
                    data = {"frame": frame, "ml_result": ml_result}
                    # 2. Push processed frame to GUI queue
                    if self.processed_queue.full():
                        try: self.processed_queue.get_nowait()
                        except queue.Empty: pass
                    self.processed_queue.put(data)
            except queue.Empty:
                pass
    
    def _draw_landmarks(self, image: MatLike, pose_results, body_angles=None):
        image.flags.writeable = True

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return cv2.flip(image, 1)

    def _infer_pose(self, image: MatLike):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.model.process(image)