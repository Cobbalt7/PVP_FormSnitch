import cv2
from cv2.typing import MatLike
import mediapipe as mp
import numpy as np
import platform
import calibration
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def infer_pose(image: MatLike, pose):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return pose.process(image)

def draw_landmarks(image: MatLike, pose_results):
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    
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

cap, cap2 = setup_cams(platform.system())

if cap2.isOpened():
    cam2_present = True
else:
    cap2.release()
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose2 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
while cap.isOpened():
    success, image = cap.read()
    if cam2_present:
        success2, image2 = cap2.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        cam1_fail = True
    elif cam2_present and not success2:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        cam2_fail = True
        
    if not cam1_fail:
        results = infer_pose(image, pose)
        
    if cam2_present and not cam2_fail:
        results2 = infer_pose(image2, pose2)
    
    if cam1_view and not cam1_fail:
        draw_landmarks(image, results)
    elif not cam1_view and not cam2_fail:
        draw_landmarks(image2, results2)
    
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