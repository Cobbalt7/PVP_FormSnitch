import threading
import cv2
import queue
import computer_vision.angles_and_evaluation as angev
import time
import csv
import os
import numpy as np
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
        self._csv_write_init()

    def run(self):
        while self.running_event.is_set():
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
                    left_knee_id = mp_pose.PoseLandmark.LEFT_KNEE.value
                    right_knee_id = mp_pose.PoseLandmark.RIGHT_KNEE.value

                    cam1_left_knee_vis = result1.pose_landmarks.landmark[left_knee_id].visibility
                    cam2_left_knee_vis = result2.pose_landmarks.landmark[left_knee_id].visibility

                    cam1_right_knee_vis = result1.pose_landmarks.landmark[right_knee_id].visibility
                    cam2_right_knee_vis = result2.pose_landmarks.landmark[right_knee_id].visibility

                    print("Cam1 L knee visibility:", cam1_left_knee_vis)
                    print("Cam2 L knee visibility:", cam2_left_knee_vis)
                    print("Cam1 R knee visibility:", cam1_right_knee_vis)
                    print("Cam2 R knee visibility:", cam2_right_knee_vis)
                    print("----------------")

                if result1.pose_landmarks and result2.pose_landmarks:                                           #
                    h, w = item1["frame"].shape[:2]                                                             # real width ir height
                    for lm1, lm2 in zip(result1.pose_landmarks.landmark, result2.pose_landmarks.landmark):      # low visibility filtras
                        if lm1.visibility < 0.5 or lm2.visibility < 0.5:                                        # 
                            points3d.append(None)                                                               #
                            continue                                                                            #
                                                                                                                #
                        points3d.append(self.calibrator.get_xyz(lm1, lm2, w, h))                                #
                else:
                    pass
                
                body_points_3d = angev.extract_points_from_triangulated_list(points3d)
                if self.calibrator.is_calibrated():
                    self._save_frame_coordinates(body_points_3d)
                print("L hip:", body_points_3d["left_hip"])                          
                print("L knee:", body_points_3d["left_knee"])
                print("L ankle:", body_points_3d["left_ankle"])
                print("R hip:", body_points_3d["right_hip"])                     #koord tikrinimui
                print("R knee:", body_points_3d["right_knee"])
                print("R ankle:", body_points_3d["right_ankle"])
                print("----------------")
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
        self.csv_file.close()
    
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

    def _csv_write_init(self):
        filename = "body_points_data.csv"
        self.joint_names = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

        file_exists = os.path.exists(filename)
        
        self.csv_file = open(filename, mode="a", newline="", buffering=1024*1024)
        self.writer = csv.writer(self.csv_file)

        if not file_exists:
            headers = []
            for joint in self.joint_names:
                headers.extend([f"{joint}_X", f"{joint}_Z", f"{joint}_Y"])
            self.writer.writerow(headers)

    def _save_frame_coordinates(self, data):
        flat_frame_coordinates = [] 
        for joint in self.joint_names:
            if data.get(joint) is not None:
                flat_frame_coordinates.extend(data[joint])
            else:
                flat_frame_coordinates.extend(["","",""])
        self.writer.writerow(flat_frame_coordinates)
