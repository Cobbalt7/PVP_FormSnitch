import threading
import cv2
import queue
import computer_vision.angles_and_evaluation as angev
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class EvalThread(threading.Thread):
    def __init__(self, data_q1, data_q2, out_q, calibrator, running_event, show_camera1_flag):
        super().__init__()
        self.data_q1 = data_q1
        self.data_q2 = data_q2
        self.out_q = out_q
        self.calibrator = calibrator
        self.running_event = running_event
        self.show_camera1 = show_camera1_flag
        self.daemon=True
        self.squat_tracker = angev.SquatTracker()

    def run(self):
        while True:
            self.running_event.wait()
            if not self.data_q1.queue or not self.data_q2.queue:
                time.sleep(0.001)
                continue
            try:
                item1 = self.data_q1.get()
                item2 = self.data_q2.get()
                points3d = []
                result1 = item1["ml_result"]
                result2 = item2["ml_result"]
                if result1.pose_landmarks and result2.pose_landmarks:
                    for lm1, lm2 in zip(result1.pose_landmarks.landmark, result2.pose_landmarks.landmark):
                        points3d.append(self.calibrator.get_xyz(lm1, lm2))
                else:
                    pass
                
                body_points_3d = angev.extract_points_from_triangulated_list(points3d)
                body_angles = None
                if angev.points_are_valid(body_points_3d):
                    body_angles = angev.calculate_body_angles(body_points_3d)
                    self._evaluate(body_angles)
                if self.show_camera1.is_set():
                    frame = item1["frame"]
                    frame = self._draw_landmarks(frame, result1)
                else:
                    frame = item2["frame"]
                    frame = self._draw_landmarks(frame, result2)
                frame = self._draw_info(frame, body_angles)

                # 2. Push processed frame to GUI queue
                if self.out_q.full():
                    try: self.out_q.get_nowait()
                    except queue.Empty: pass
                self.out_q.put(frame)

            except queue.Empty:
                pass
    
    def _evaluate(self, body_angles):
        feedback = self.squat_tracker.update(body_angles)

        left_knee = body_angles["left_knee_angle"]
        right_knee = body_angles["right_knee_angle"]
        left_hip = body_angles["left_hip_angle"]
        right_hip = body_angles["right_hip_angle"]

        print("Kairio kelio kampas:", left_knee)
        print("Desinio kelio kampas:", right_knee)
        print("Kairio klubo kampas:", left_hip)
        print("Desinio klubo kampas:", right_hip)

        if feedback is not None:
            print("---------------")
            print(feedback["message"])
            print("Pritūpimų skaičius:", feedback["squat_count"])
            print("Minimalus kelio kampas:", feedback["min_knee_angle"])
            print("Minimalus klubo kampas:", feedback["min_hip_angle"])
            print("---------------")
            
    def _draw_landmarks(self, image, pose_results):
        image.flags.writeable = True

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return cv2.flip(image, 1)
    
    def _draw_info(self, frame, body_angles=None):
        if body_angles is not None:
            left_knee = body_angles["left_knee_angle"]
            right_knee = body_angles["right_knee_angle"]
            left_hip = body_angles["left_hip_angle"]
            right_hip = body_angles["right_hip_angle"]

            if left_knee is not None:
                cv2.putText(frame, f"L knee: {left_knee:.1f}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            if right_knee is not None:
                cv2.putText(frame, f"R knee: {right_knee:.1f}",
                            (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            if left_hip is not None:
                cv2.putText(frame, f"L hip: {left_hip:.1f}",
                            (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            if right_hip is not None:
                cv2.putText(frame, f"R hip: {right_hip:.1f}",
                            (30, 145), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
        return frame
