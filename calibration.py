import cv2
from cv2 import VideoCapture
import numpy as np
import time

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 8)

def detect_chessboard(im):
   corners = []
   # Find the chess board corners
   # If desired number of corners are
   # found in the image then ret = true
   ret, corners = cv2.findChessboardCorners(
                      im, CHECKERBOARD, 
                      cv2.CALIB_CB_ADAPTIVE_THRESH 
                      + cv2.CALIB_CB_FAST_CHECK + 
                      cv2.CALIB_CB_NORMALIZE_IMAGE)
   return ret, corners

def parse_images(images1, images2, obj3d, imgcnt):
   points3d = []
   points2d1 = []
   points2d2 = []
   criteria = (cv2.TERM_CRITERIA_EPS + 
               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   for i in range(0, imgcnt):
      grayColor1 = cv2.cvtColor(images1[i], cv2.COLOR_BGR2GRAY)
      ret1, corners1 = detect_chessboard(grayColor1);
      grayColor2 = cv2.cvtColor(images2[i], cv2.COLOR_BGR2GRAY)
      ret2, corners2 = detect_chessboard(grayColor2);

      # If desired number of corners can be detected then,
      # refine the pixel coordinates and display
      # them on the images of checker board
      if ret1 and ret2:
         points3d.append(obj3d)

         # Refining pixel coordinates
         # for given 2d points.
         corners1 = cv2.cornerSubPix(
            grayColor1, corners1, (11, 11), (-1, -1), criteria)
         corners2 = cv2.cornerSubPix(
            grayColor2, corners2, (11, 11), (-1, -1), criteria)

         points2d1.append(corners1)
         points2d2.append(corners2)

         ## Draw and display the corners
         #image = cv2.drawChessboardCorners(image, 
         #                                     CHECKERBOARD, 
         #                                     corners2, ret)
   return points3d, points2d1, points2d2

def calibrate_cameras(cam1: VideoCapture, cam2: VideoCapture):
   # stop the iteration when specified
   # accuracy, epsilon, is reached or
   # specified number of iterations are completed.

   # Vector for 3D points
   threedpoints = []

   #  3D points real world coordinates
   objectp3d = np.zeros((1, CHECKERBOARD[0] 
                         * CHECKERBOARD[1], 
                         3), np.float32)
   objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                  0:CHECKERBOARD[1]].T.reshape(-1, 2)
   prev_img_shape = None


   # Extracting path of individual image stored
   # in a given directory. Since no path is
   # specified, it will take current directory
   # jpg files alone
   images1 = []
   images2 = []
   for i in range(0,30):
      time.sleep(0.030)
      ret, frame = cam1.read()
      if ret:
         images1.append(frame)
      ret, frame = cam2.read()
      if ret:
         images2.append(frame)
   threedpoints, twodpoints1, twodpoints2 = parse_images(images1,images2, objectp3d, 30);
   
   #h, w = image.shape[:2]
   
   # Perform camera calibration by
   # passing the value of above found out 3D points (threedpoints)
   # and its corresponding pixel coordinates of the
   # detected corners (twodpoints)
   h, w = images1[0].shape[:2]
   image_size = (w, h)
   ret1, matrix1, distortion1, r_vecs1, t_vecs1 = cv2.calibrateCamera(
       threedpoints, twodpoints1, image_size, None, None)

   ret2, matrix2, distortion2, r_vecs2, t_vecs2 = cv2.calibrateCamera(
       threedpoints, twodpoints2, image_size, None, None)
   
   stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
   
   ret, matrix1, distortion1, matrix2, distortion2, R, T, E, F = cv2.stereoCalibrate(
      threedpoints, twodpoints1, twodpoints2, matrix1, distortion1, 
      matrix2, distortion2, image_size, criteria = criteria, flags = stereocalibration_flags)
   rectify_flags = 0
   R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
       matrix1, distortion1, matrix2, distortion2, 
       image_size, R, T, rectify_flags, alpha=0)
   return P1, P2