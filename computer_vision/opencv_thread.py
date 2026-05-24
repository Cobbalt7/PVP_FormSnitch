import threading
import cv2
import queue
import mediapipe as mp
import computer_vision.angles_and_evaluation as angev
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
                    data = {"frame": frame, "ml_result": ml_result}
                    # 2. Push processed frame to GUI queue
                    if self.processed_queue.full():
                        try: self.processed_queue.get_nowait()
                        except queue.Empty: pass
                    self.processed_queue.put(data)
            except queue.Empty:
                pass

    def _infer_pose(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.model.process(image)