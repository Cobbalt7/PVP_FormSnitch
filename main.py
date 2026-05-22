import cv2
from cv2.typing import MatLike
import mediapipe as mp
import numpy as np
import platform
import calibration
import angles_and_evaluation as angev
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def infer_pose(image: MatLike, pose):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pose.process(image)

def draw_landmarks(image: MatLike, pose_results, body_angles=None):
    image.flags.writeable = True

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    display_image = cv2.flip(image, 1)

    if body_angles is not None:
        left_knee = body_angles["left_knee_angle"]
        right_knee = body_angles["right_knee_angle"]
        left_hip = body_angles["left_hip_angle"]
        right_hip = body_angles["right_hip_angle"]

        if left_knee is not None:
            cv2.putText(display_image, f"L knee: {left_knee:.1f}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        if right_knee is not None:
            cv2.putText(display_image, f"R knee: {right_knee:.1f}",
                        (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        if left_hip is not None:
            cv2.putText(display_image, f"L hip: {left_hip:.1f}",
                        (30, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        if right_hip is not None:
            cv2.putText(display_image, f"R hip: {right_hip:.1f}",
                        (30, 145), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Pose', display_image)
    
def setup_cams(platform):
    match platform:
        case 'Linux':
            capture_arg = cv2.CAP_V4L2
            second_cam_num = 2
        case 'Windows':
            capture_arg = cv2.CAP_DSHOW
            second_cam_num = 1
        case _:
            capture_arg = cv2.CAP_V4L2
    cap = cv2.VideoCapture(0, capture_arg)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap2 = cv2.VideoCapture(second_cam_num, capture_arg)
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return cap, cap2
    

cam2_present = False
cam1_fail = False
cam2_fail = False
cam1_view = True
proj_mat =np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]], dtype=np.float32)
proj_mat2 = proj_mat.copy()
points3d=[]

cap, cap2 = setup_cams(platform.system())

if cap2.isOpened():
    cam2_present = True
else:
    cap2.release()
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose2 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

squat_tracker = angev.SquatTracker()

while cap.isOpened():
    body_angles = None
    success, image = cap.read()

    if cam2_present:
        success2, image2 = cap2.read()

    if not success:
        print("Ignoring empty camera frame.")
        cam1_fail = True

    elif cam2_present and not success2:
        print("Ignoring empty camera frame.")
        cam2_fail = True

    if not cam1_fail:
        results = infer_pose(image, pose)

    if cam2_present and not cam2_fail:
        results2 = infer_pose(image2, pose2)

    if not cam1_fail and cam2_present and not cam2_fail:
        points3d = []

        if results.pose_landmarks and results2.pose_landmarks:
            image_height, image_width = image.shape[:2]

            for lm1, lm2 in zip(results.pose_landmarks.landmark, results2.pose_landmarks.landmark):
                point = calibration.get_point_image_coords(lm1, image_width, image_height)
                point2 = calibration.get_point_image_coords(lm2, image_width, image_height)

                point_3d = calibration.get_xyz(proj_mat, proj_mat2, point, point2)
                points3d.append(point_3d)

            body_points_3d = angev.extract_points_from_triangulated_list(points3d)

            if angev.points_are_valid(body_points_3d):
                body_angles = angev.calculate_body_angles(body_points_3d)

                feedback = squat_tracker.update(body_angles)

                left_knee = body_angles["left_knee_angle"]
                right_knee = body_angles["right_knee_angle"]
                left_hip = body_angles["left_hip_angle"]
                right_hip = body_angles["right_hip_angle"]

                print("Kairio kelio kampas:", left_knee)
                print("Desinio kelio kampas:", right_knee)
                print("Kairio klubo kampas:", left_hip)
                print("Desinio klubo kampas:", right_hip)

                if feedback is not None:
                    image.flags.writeable = True
                    print("---------------")
                    print(feedback["message"])
                    print("Pritūpimų skaičius:", feedback["squat_count"])
                    print("Minimalus kelio kampas:", feedback["min_knee_angle"])
                    print("Minimalus klubo kampas:", feedback["min_hip_angle"])
                    print("---------------")

    if cam1_view and not cam1_fail:
        draw_landmarks(image, results, body_angles)

    elif not cam1_view and not cam2_fail:
        draw_landmarks(image2, results2, body_angles)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    elif key == ord('s') and cam2_present:
        cam1_view = not cam1_view

    elif key == ord('c') and cam2_present:
        proj_mat, proj_mat2 = calibration.calibrate_cameras(cap, cap2)

    cam1_fail = False
    cam2_fail = False

cap.release()