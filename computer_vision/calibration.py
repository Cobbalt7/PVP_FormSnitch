import time
import cv2
from cv2 import VideoCapture
import numpy as np
import threading
import queue

CALIBRATION_FILE = "stereo_calibration.npz"
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
        
    def calibrate(self, frame_q1, frame_q2, progress_callback=None) -> bool:
        """
        Captures frames from both cameras, runs stereo calibration,
        and saves coefficients internally. Returns True if successful.
        """
        square_size = 33.0
        # 3D points real world coordinates setup
        objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)*square_size

        images1 = []
        images2 = []
        total_images = 30

        while len(images1) < total_images:
            try:
                images1.append(frame_q1.get(timeout=1.0))
                images2.append(frame_q2.get(timeout=1.0))

                if progress_callback is not None:
                    progress_callback(len(images1), total_images)

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
            self.save_calibration()
            self.verify_saved_calibration()
            print("Calibration completed and stored successfully!")
            return True

    def save_calibration(self, filename=CALIBRATION_FILE):
        """
        Išsaugo stereo kalibracijos rezultatus į failą.
        Tai leidžia kitą kartą naudoti tą pačią kamerų padėtį be naujos kalibracijos.
        """
        if not self.is_calibrated():
            print("Calibration was not saved because calibrator is not calibrated.")
            return False

        np.savez(
            filename,
            matrix1=self.calibration.mat[0],
            matrix2=self.calibration.mat[1],
            distortion1=self.calibration.dist[0],
            distortion2=self.calibration.dist[1],
            rotation1=self.calibration.rot[0],
            rotation2=self.calibration.rot[1],
            projection1=self.calibration.proj[0],
            projection2=self.calibration.proj[1],
        )

        print(f"Calibration saved to {filename}")
        return True

    def load_calibration(self, filename=CALIBRATION_FILE):
        """
        Nuskaito stereo kalibracijos rezultatus iš failo.
        Naudojama tada, kai kameros po kalibracijos nebuvo pajudintos.
        """
        try:
            data = np.load(filename)

            matrix1 = data["matrix1"]
            matrix2 = data["matrix2"]
            distortion1 = data["distortion1"]
            distortion2 = data["distortion2"]
            rotation1 = data["rotation1"]
            rotation2 = data["rotation2"]
            projection1 = data["projection1"]
            projection2 = data["projection2"]

            self.calibration = CalibrationResult(
                (matrix1, matrix2),
                (distortion1, distortion2),
                (rotation1, rotation2),
                (projection1, projection2)
            )

            print(f"Calibration loaded from {filename}")
            return True

        except FileNotFoundError:
            print(f"Calibration file {filename} not found.")
            return False

        except Exception as e:
            print("Failed to load calibration:", e)
            return False
    
    def verify_saved_calibration(self, filename=CALIBRATION_FILE):
        """
        Patikrina, ar išsaugotos ir vėl nuskaitytos kalibracijos matricos sutampa
        su šiuo metu atmintyje esančiomis matricomis.
        """
        if not self.is_calibrated():
            print("Cannot verify: calibrator is not calibrated.")
            return False

        try:
            data = np.load(filename)

            checks = {
                "matrix1": (self.calibration.mat[0], data["matrix1"]),
                "matrix2": (self.calibration.mat[1], data["matrix2"]),
                "distortion1": (self.calibration.dist[0], data["distortion1"]),
                "distortion2": (self.calibration.dist[1], data["distortion2"]),
                "rotation1": (self.calibration.rot[0], data["rotation1"]),
                "rotation2": (self.calibration.rot[1], data["rotation2"]),
                "projection1": (self.calibration.proj[0], data["projection1"]),
                "projection2": (self.calibration.proj[1], data["projection2"]),
            }

            all_ok = True

            print("--- Saved calibration verification ---")

            for name, (original, loaded) in checks.items():
                exact_equal = np.array_equal(original, loaded)
                close_equal = np.allclose(original, loaded)

                max_diff = np.max(np.abs(original - loaded))

                print(f"{name}: exact={exact_equal}, allclose={close_equal}, max_diff={max_diff}")

                if not close_equal:
                    all_ok = False

            if all_ok:
                print("Calibration save/load verification PASSED.")
            else:
                print("Calibration save/load verification FAILED.")

            return all_ok

        except Exception as e:
            print("Failed to verify saved calibration:", e)
            return False

    def get_xyz(self, lm1, lm2, image_width=480, image_height=640) -> np.ndarray:                                 # cia irasom w ir h
        """
        Triangulates a single pair of 2D points into a 3D coordinate using stored coefficients.
        pt1: (x, y) pixel coordinate from Camera 1
        pt2: (x, y) pixel coordinate from Camera 2
        """
        with self._lock:
            #print(f"MediaPipe X/Y: ({lm1.x:.3f}, {lm1.y:.3f})")
            # Vietoje fiksuotų 480x640 reikšmių naudojamas realus kadro dydis.
            pt1 = self._get_point_image_coords(lm1, image_width, image_height)                                    # tada cia naudojam
            pt2 = self._get_point_image_coords(lm2, image_width, image_height)                                    # h, w = item1["frame"].shape[:2]  ir  points3d.append(self.calibrator.get_xyz(lm1, lm2, w, h))  duoda realius pixels is evaluation_thread
            #print(f"Pixel X/Y: {pt1}")
            if self.is_calibrated():
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
            #print(f"Rectified Pixel X/Y: {rectified_pt1}")

            # Po rektifikacijos tas pats taškas abiejose kamerose turėtų būti panašiame Y lygyje.
            # Jei Y skirtumas per didelis, taškas laikomas nepatikimu ir netrianguliuojamas.
            #y_diff = abs(rectified_pt1[1] - rectified_pt2[1])

            #if y_diff > 30:                 #filtras
            #   return None

            # OpenCV expects float32 arrays of shape (2, N) for 2D points
            points1 = np.array([rectified_pt1], dtype=np.float32).T  # Shape: (2, 1)
            points2 = np.array([rectified_pt2], dtype=np.float32).T  # Shape: (2, 1)

            try:
                points_4d = cv2.triangulatePoints(self.calibration.proj[0], self.calibration.proj[1], points1, points2)
            except Exception:
                print("Error: Triangulation failed!")
                return None

            # Convert from homogeneous (X, Y, Z, W) to Cartesian (x, y, z)
            points_3d = points_4d[:3, :] / points_4d[3, :]
            coord_3d = points_3d.T[0]

            if not np.all(np.isfinite(coord_3d)):
                return None

            if np.linalg.norm(coord_3d) > 10000:
                return None

            return coord_3d
    
    def is_calibrated(self):
        return self.calibration.calibrated

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
        pt = np.array([[pixel_pt]], dtype=np.float32)

        rectified_pt = cv2.undistortPoints(
            pt,
            camera_matrix,
            dist_coeffs,
            R=R_matrix,
            P=P_matrix
        )

        return rectified_pt[0][0]

    @staticmethod
    def _get_point_image_coords(pt, im_w, im_h):
        pixel_x = float(pt.x * im_w)
        pixel_y = float(pt.y * im_h)
        return (pixel_x, pixel_y)
