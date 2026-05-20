import cv2
from cv2 import VideoCapture
import numpy as np
import time

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 8)

#matrix1, distortion1 = None, None
#matrix2, distortion2 = None, None
#R1, R2, P1, P2 = None, None, None, None
calibrated = False

def _evaluate_calibration(threedpoints, twodpoints1, twodpoints2, matrix1, distortion1, matrix2, distortion2, r_vecs1, t_vecs1, r_vecs2, t_vecs2):
   """
   Calculates and prints the mean reprojection error for both cameras.
   """
   mean_error_cam1 = 0
   mean_error_cam2 = 0
   total_points = 0

   for i in range(len(threedpoints)):
      # --- Camera 1 ---
      # Project the 3D points to 2D using the calibrated parameters
      imgpoints1_projected, _ = cv2.projectPoints(
          threedpoints[i], r_vecs1[i], t_vecs1[i], matrix1, distortion1
      )
      # Calculate the absolute norm (distance) between detected and projected points
      error1 = cv2.norm(twodpoints1[i], imgpoints1_projected, cv2.NORM_L2) / len(imgpoints1_projected)
      mean_error_cam1 += error1

      # --- Camera 2 ---
      imgpoints2_projected, _ = cv2.projectPoints(
          threedpoints[i], r_vecs2[i], t_vecs2[i], matrix2, distortion2
      )
      error2 = cv2.norm(twodpoints2[i], imgpoints2_projected, cv2.NORM_L2) / len(imgpoints2_projected)
      mean_error_cam2 += error2

      total_points += 1

   print(f"--- Calibration Evaluation ---")
   print(f"Total pairs of images used: {total_points}")
   print(f"Camera 1 Mean Reprojection Error: {mean_error_cam1 / total_points:.4f} pixels")
   print(f"Camera 2 Mean Reprojection Error: {mean_error_cam2 / total_points:.4f} pixels")

def _detect_chessboard(im):
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

def _parse_images(images1, images2, obj3d, imgcnt):
   points3d = []
   points2d1 = []
   points2d2 = []
   criteria = (cv2.TERM_CRITERIA_EPS + 
               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
   for i in range(0, imgcnt):
      grayColor1 = cv2.cvtColor(images1[i], cv2.COLOR_BGR2GRAY)
      ret1, corners1 = _detect_chessboard(grayColor1);
      grayColor2 = cv2.cvtColor(images2[i], cv2.COLOR_BGR2GRAY)
      ret2, corners2 = _detect_chessboard(grayColor2);

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
      ret, frame = cam1.read()
      if ret:
         images1.append(frame)
      ret, frame = cam2.read()
      if ret:
         images2.append(frame)
   threedpoints, twodpoints1, twodpoints2 = _parse_images(images1,images2, objectp3d, 30);
   
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
   
   _evaluate_calibration(threedpoints, twodpoints1, twodpoints2, 
                        matrix1, distortion1, matrix2, distortion2, 
                        r_vecs1, t_vecs1, r_vecs2, t_vecs2)
   
   rectify_flags = 0
   R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
       matrix1, distortion1, matrix2, distortion2, 
       image_size, R, T, rectify_flags, alpha=0)
   
   calibrated = True
   return P1, P2

def _rectify_point(pixel_pt, camera_matrix, dist_coeffs, R_matrix, P_matrix):
   """
   Takes a raw pixel coordinate (u, v) from an uncalibrated frame and 
   transforms it into the rectified camera coordinate space.
   """
   # Reshape point to format expected by OpenCV: (1, 1, 2)
   pt = np.array([[pixel_pt]], dtype=np.float32)
   
   # Undistort and apply rectification rotation
   rectified_pt = cv2.undistortImagePoints(pt, camera_matrix, dist_coeffs, R_matrix, P_matrix)
    
   # Return as a clean (x, y) tuple
   return rectified_pt[0][0]

def get_xyz(P1, P2, pt1, pt2):
   """
   Triangulates a single pair of 2D points into a 3D coordinate.
   
   P1, P2: Projection matrices from stereoRectify (3x4)
   pt1: (x, y) pixel coordinate from Camera 1
   pt2: (x, y) pixel coordinate from Camera 2
   """
   if calibrated:
      rectified_pt1 = _rectify_point(pt1, matrix1, distortion1, R1, P1)
      rectified_pt2 = _rectify_point(pt2, matrix2, distortion2, R2, P2)
   else:
      #bypass if uncalibrated
      rectified_pt1 = pt1
      rectified_pt2 = pt2
      
   # OpenCV expects float32 arrays of shape (2, N) for 2D points
   points1 = np.array([rectified_pt1], dtype=np.float32).T  # Shape: (2, 1)
   points2 = np.array([rectified_pt2], dtype=np.float32).T  # Shape: (2, 1)
   
   # Triangulate points outputs a 4x1 homogeneous vector
   points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
   
   # Convert from homogeneous (X, Y, Z, W) to Cartesian (x, y, z)
   # Divide X, Y, Z by W
   points_3d = points_4d[:3, :] / points_4d[3, :]
   
   # Reshape to a clean 1D coordinate array [x, y, z]
   coord_3d = points_3d.T[0]
   
   return coord_3d

def get_point_image_coords(pt, im_w, im_h):
   # Convert normalized coordinates to pixel coordinates
   pixel_x = int(pt.x * im_w)
   pixel_y = int(pt.y * im_h)
   return (pixel_x, pixel_y)