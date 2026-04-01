import time
import cv2
import platform
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision

import os

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
      connection_drawing_spec=pose_connection_style)

   return annotated_image

class landmarker_and_result():
   def __init__(self):
      self.result = None
      self.landmarker = mp.tasks.vision.PoseLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      options = mp.tasks.vision.PoseLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_full.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         min_pose_detection_confidence = 0.5, # lower than value to get predictions more often
         min_pose_presence_confidence = 0.5, # lower than value to get predictions more often
         min_tracking_confidence = 0.5, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def main():
   #test_indices()
    # Setting which camera to display
   camera1 = True
   # Force Qt to use X11/XWayland
   if platform.system() == 'Linux':
      os.environ["QT_QPA_PLATFORM"] = "xcb"
      # access webcam
      cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
      cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
      cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
      cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

      cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
      cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
      cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
      cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
      
   elif platform.system() == 'Windows':
      # access webcam
      cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
      cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
      cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
      cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

      cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
      cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
      cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
      cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   # create landmarker
   pose_landmarker1 = landmarker_and_result()
   pose_landmarker2 = landmarker_and_result()
   
   
   # Warm up cameras
   for _ in range(5):
      cap1.read()
      cap2.read()
   
   while True:# pull frame
      ret1, frame1 = cap1.read()
      if ret1:
         # mirror frame
         frame1 = cv2.flip(frame1, 1)
         # update landmarker results
         pose_landmarker1.detect_async(frame1)
      
      ret2, frame2 = cap2.read()
      if ret2:
         frame2 = cv2.flip(frame2, 1)
         # update landmarker results
         pose_landmarker2.detect_async(frame2)
      
      if camera1 and ret1:
         #print(pose_landmarker1.result)
         if pose_landmarker1.result is not None:
           frame1 = draw_landmarks_on_image(frame1, pose_landmarker1.result)
           # display frame
           cv2.imshow('frame',frame1)
      elif ret2:
         #print(pose_landmarker2.result)
         if pose_landmarker2.result is not None:
           frame2 = draw_landmarks_on_image(frame2, pose_landmarker2.result)
         # display frame
         cv2.imshow('frame',frame2)
      key = cv2.waitKey(1)
      if key == ord('s'):
         camera1 = not camera1
      elif key == ord('q'):
          break
        
   # release everything
   pose_landmarker1.close()
   cap1.release()
   
   pose_landmarker2.close()
   cap2.release()
   
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()