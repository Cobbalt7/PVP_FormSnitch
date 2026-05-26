import time
import cv2
from cv2 import VideoCapture
import numpy as np
import threading
import queue

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 8)


class CalibrationResult:
    def __init__(self, matrix=None, distortion=None, rotation=None, projection=None):
        if matrix is not None:
            self.mat = matrix
            self.dist = distortion
            self.rot = rotation
            self.proj = projection
            self.calibrated = True
        else:
            self.mat = (None, None)
            self.dist = (None, None)
            self.rot = (None, None)
            self.proj = (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
            )
            self.calibrated = False


class Calibrator:
    def __init__(self):
        # Stores the coefficients once calibrate() runs successfully
        self.calibration = CalibrationResult()
        self._lock = threading.RLock()
        
    def calibrate(self, frame_q1, frame_q2) -> bool:
        """
        Captures frames from both cameras, runs stereo calibration,
        and saves coefficients internally. Returns True if successful.
        """
        # 3D points real world coordinates setup
        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

        images1 = []
        images2 = []
        while len(images1) < 30:
            try:
                images1.append(frame_q1.get(timeout=1.0))
                images2.append(frame_q2.get(timeout=1.0))
            except queue.Empty:
                time.sleep(0.001)
                
        with self._lock:
            threedpoints, twodpoints1, twodpoints2 = self._parse_images(images1, images2, objectp3d, 30)

            if not threedpoints:
                print("Error: No checkerboards detected in captured frames.")
                return False

            h, w = images1[0].shape[:2]
            image_size = (w, h)

            # Individual Camera Calibration
            try:
                ret1, matrix1, distortion1, r_vecs1, t_vecs1 = cv2.calibrateCamera(
                    threedpoints, twodpoints1, image_size, None, None
                )
            except Exception:
                print("Error: Cam1 calibration failed!")
                return False

            try:
                ret2, matrix2, distortion2, r_vecs2, t_vecs2 = cv2.calibrateCamera(
                    threedpoints, twodpoints2, image_size, None, None
                )
            except Exception:
                print("Error: Cam2 calibration failed!")
                return False

            # Stereo Calibration
            stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            try:
                ret, matrix1, distortion1, matrix2, distortion2, R, T, E, F = cv2.stereoCalibrate(
                    threedpoints, twodpoints1, twodpoints2, matrix1, distortion1,
                    matrix2, distortion2, image_size, criteria=criteria, flags=stereocalibration_flags
                )
            except Exception:
                print("Error: Stereo calibration failed!")
                return False

            self._evaluate_calibration(
                threedpoints, twodpoints1, twodpoints2,
                matrix1, distortion1, matrix2, distortion2,
                r_vecs1, t_vecs1, r_vecs2, t_vecs2
            )

            # Stereo Rectification
            rectify_flags = 0
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                matrix1, distortion1, matrix2, distortion2,
                image_size, R, T, rectify_flags, alpha=0
            )

            # Save results to instance property
            self.calibration = CalibrationResult((matrix1, matrix2), (distortion1, distortion2), (R1, R2), (P1, P2))
            print("Calibration completed and stored successfully!")
            return True

    def get_xyz(self, lm1, lm2) -> np.ndarray:
        """
        Triangulates a single pair of 2D points into a 3D coordinate using stored coefficients.
        pt1: (x, y) pixel coordinate from Camera 1
        pt2: (x, y) pixel coordinate from Camera 2
        """
        with self._lock:
            pt1 = self._get_point_image_coords(lm1, 640, 480)
            pt2 = self._get_point_image_coords(lm2, 640, 480)

            if self.calibration.calibrated:
                rectified_pt1 = self._rectify_point(
                    pt1, self.calibration.mat[0], self.calibration.dist[0], self.calibration.rot[0], self.calibration.proj[0]
                )
                rectified_pt2 = self._rectify_point(
                    pt2, self.calibration.mat[1], self.calibration.dist[1], self.calibration.rot[1], self.calibration.proj[1]
                )
            else:
                # Bypass rectification if uncalibrated
                rectified_pt1 = pt1
                rectified_pt2 = pt2

            # OpenCV expects float32 arrays of shape (2, N) for 2D points
            points1 = np.array([rectified_pt1], dtype=np.float32).T  # Shape: (2, 1)
            points2 = np.array([rectified_pt2], dtype=np.float32).T  # Shape: (2, 1)

            try:
                points_4d = cv2.triangulatePoints(self.calibration.proj[0], self.calibration.proj[1], points1, points2)
            except Exception:
                print("Error: Triangulation failed!")
                return np.array([0, 0, 0], dtype=np.float32)

            # Convert from homogeneous (X, Y, Z, W) to Cartesian (x, y, z)
            points_3d = points_4d[:3, :] / points_4d[3, :]
            coord_3d = points_3d.T[0]

            return coord_3d

    # --- PRIVATE INTERNAL METHODS ---

    def _detect_chessboard(self, im):
        ret, corners = cv2.findChessboardCorners(
            im, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        return ret, corners

    def _parse_images(self, images1, images2, obj3d, imgcnt):
        points3d = []
        points2d1 = []
        points2d2 = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for i in range(min(len(images1), len(images2), imgcnt)):
            grayColor1 = cv2.cvtColor(images1[i], cv2.COLOR_BGR2GRAY)
            ret1, corners1 = self._detect_chessboard(grayColor1)
            grayColor2 = cv2.cvtColor(images2[i], cv2.COLOR_BGR2GRAY)
            ret2, corners2 = self._detect_chessboard(grayColor2)

            if ret1 and ret2:
                points3d.append(obj3d)
                corners1 = cv2.cornerSubPix(grayColor1, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(grayColor2, corners2, (11, 11), (-1, -1), criteria)
                points2d1.append(corners1)
                points2d2.append(corners2)
                
        return points3d, points2d1, points2d2

    def _evaluate_calibration(self, threedpoints, twodpoints1, twodpoints2, 
                              matrix1, distortion1, matrix2, distortion2, 
                              r_vecs1, t_vecs1, r_vecs2, t_vecs2):
        mean_error_cam1 = 0
        mean_error_cam2 = 0
        total_points = 0

        for i in range(len(threedpoints)):
            imgpoints1_projected, _ = cv2.projectPoints(threedpoints[i], r_vecs1[i], t_vecs1[i], matrix1, distortion1)
            error1 = cv2.norm(twodpoints1[i], imgpoints1_projected, cv2.NORM_L2) / len(imgpoints1_projected)
            mean_error_cam1 += error1

            imgpoints2_projected, _ = cv2.projectPoints(threedpoints[i], r_vecs2[i], t_vecs2[i], matrix2, distortion2)
            error2 = cv2.norm(twodpoints2[i], imgpoints2_projected, cv2.NORM_L2) / len(imgpoints2_projected)
            mean_error_cam2 += error2
            total_points += 1

        print("--- Calibration Evaluation ---")
        print(f"Total pairs of images used: {total_points}")
        print(f"Camera 1 Mean Reprojection Error: {mean_error_cam1 / max(1, total_points):.4f} pixels")
        print(f"Camera 2 Mean Reprojection Error: {mean_error_cam2 / max(1, total_points):.4f} pixels")

    def _rectify_point(self, pixel_pt, camera_matrix, dist_coeffs, R_matrix, P_matrix):
        pt = np.array([[pixel_pt]], dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rectified_pt = cv2.undistortImagePoints(pt, camera_matrix, dist_coeffs, criteria)
        return rectified_pt[0][0]

    @staticmethod
    def _get_point_image_coords(pt, im_w, im_h):
        pixel_x = int(pt.x * im_w)
        pixel_y = int(pt.y * im_h)
        return (pixel_x, pixel_y)
