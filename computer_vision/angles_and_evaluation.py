import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


def get_point_2d_as_3d(pose_results, landmark_name, image_width, image_height, min_visibility=0.5):
    """
    Iš MediaPipe landmark paima 2D tašką ir paverčia į 3D tašką su z = 0.

    Grąžina:
    [x, y, 0] arba None, jei taškas blogai matomas.
    """

    if pose_results.pose_landmarks is None:
        return None

    landmark_id = getattr(mp_pose.PoseLandmark, landmark_name).value
    landmark = pose_results.pose_landmarks.landmark[landmark_id]

    if landmark.visibility < min_visibility:
        return None

    x = landmark.x * image_width
    y = landmark.y * image_height
    z = 0

    return [x, y, z]


def get_point_from_list(points3d, landmark):
    index = landmark.value

    if points3d is None:
        return None

    if len(points3d) <= index:
        return None

    return points3d[index]


def extract_points_from_triangulated_list(points3d):
    """
    Iš jau trianguliuotų MediaPipe taškų sąrašo paima reikalingus taškus kampams.
    points3d turi būti sąrašas, kuriame taškai sudėti ta pačia eilės tvarka kaip MediaPipe landmark'ai.
    """

    return {
        "left_shoulder": get_point_from_list(points3d, mp_pose.PoseLandmark.LEFT_SHOULDER),
        "right_shoulder": get_point_from_list(points3d, mp_pose.PoseLandmark.RIGHT_SHOULDER),

        "left_hip": get_point_from_list(points3d, mp_pose.PoseLandmark.LEFT_HIP),
        "right_hip": get_point_from_list(points3d, mp_pose.PoseLandmark.RIGHT_HIP),

        "left_knee": get_point_from_list(points3d, mp_pose.PoseLandmark.LEFT_KNEE),
        "right_knee": get_point_from_list(points3d, mp_pose.PoseLandmark.RIGHT_KNEE),

        "left_ankle": get_point_from_list(points3d, mp_pose.PoseLandmark.LEFT_ANKLE),
        "right_ankle": get_point_from_list(points3d, mp_pose.PoseLandmark.RIGHT_ANKLE),
    }


def extract_points_from_mediapipe_2d(pose_results, image_width, image_height):
    """
    Iš MediaPipe rezultatų paima reikalingus taškus kampų skaičiavimui.
    Naudojama testavimui be trianguliacijos.
    """

    points_3d = {
        "left_shoulder": get_point_2d_as_3d(
            pose_results, "LEFT_SHOULDER", image_width, image_height
        ),
        "right_shoulder": get_point_2d_as_3d(
            pose_results, "RIGHT_SHOULDER", image_width, image_height
        ),

        "left_hip": get_point_2d_as_3d(
            pose_results, "LEFT_HIP", image_width, image_height
        ),
        "right_hip": get_point_2d_as_3d(
            pose_results, "RIGHT_HIP", image_width, image_height
        ),

        "left_knee": get_point_2d_as_3d(
            pose_results, "LEFT_KNEE", image_width, image_height
        ),
        "right_knee": get_point_2d_as_3d(
            pose_results, "RIGHT_KNEE", image_width, image_height
        ),

        "left_ankle": get_point_2d_as_3d(
            pose_results, "LEFT_ANKLE", image_width, image_height
        ),
        "right_ankle": get_point_2d_as_3d(
            pose_results, "RIGHT_ANKLE", image_width, image_height
        )
    }

    return points_3d


def knee_points_are_valid(points_3d, side):
    """
    Patikrina, ar konkrečios pusės kelio kampui yra reikalingi taškai:
    klubas, kelis ir čiurna.
    """

    return (
        points_3d.get(f"{side}_hip") is not None and
        points_3d.get(f"{side}_knee") is not None and
        points_3d.get(f"{side}_ankle") is not None
    )


def hip_points_are_valid(points_3d, side):
    """
    Patikrina, ar konkrečios pusės klubo kampui yra reikalingi taškai:
    petys, klubas ir kelis.
    """

    return (
        points_3d.get(f"{side}_shoulder") is not None and
        points_3d.get(f"{side}_hip") is not None and
        points_3d.get(f"{side}_knee") is not None
    )


def points_are_valid(points_3d):
    """
    Patikrina, ar yra bent viena matoma kūno pusė,
    iš kurios galima apskaičiuoti bent kelio arba klubo kampą.
    """

    if points_3d is None:
        return False

    left_knee_valid = knee_points_are_valid(points_3d, "left")
    right_knee_valid = knee_points_are_valid(points_3d, "right")

    left_hip_valid = hip_points_are_valid(points_3d, "left")
    right_hip_valid = hip_points_are_valid(points_3d, "right")

    return (
        left_knee_valid or
        right_knee_valid or
        left_hip_valid or
        right_hip_valid
    )



def calculate_angle_3d(a, b, c):
    """
    Apskaičiuoja kampą tarp trijų 3D taškų.
    Kampas skaičiuojamas taške b.
    https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    """

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)

    # Apsauga nuo skaičiavimo paklaidų
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_body_angles(points_3d):
    """
    Apskaičiuoja pagrindinius pritūpimo kampus iš matomų taškų.
    Jei matoma tik viena kūno pusė, naudojama tik ji.
    Jei matomos abi pusės, skaičiuojamas vidurkis.
    """

    angles = {}

    angles["left_knee_angle"] = None
    angles["right_knee_angle"] = None
    angles["left_hip_angle"] = None
    angles["right_hip_angle"] = None

    # Kairės kojos kelio kampas
    if knee_points_are_valid(points_3d, "left"):
        angles["left_knee_angle"] = calculate_angle_3d(
            points_3d["left_hip"],
            points_3d["left_knee"],
            points_3d["left_ankle"]
        )

    # Dešinės kojos kelio kampas
    if knee_points_are_valid(points_3d, "right"):
        angles["right_knee_angle"] = calculate_angle_3d(
            points_3d["right_hip"],
            points_3d["right_knee"],
            points_3d["right_ankle"]
        )

    # Kairės pusės klubo kampas
    if hip_points_are_valid(points_3d, "left"):
        angles["left_hip_angle"] = calculate_angle_3d(
            points_3d["left_shoulder"],
            points_3d["left_hip"],
            points_3d["left_knee"]
        )

    # Dešinės pusės klubo kampas
    if hip_points_are_valid(points_3d, "right"):
        angles["right_hip_angle"] = calculate_angle_3d(
            points_3d["right_shoulder"],
            points_3d["right_hip"],
            points_3d["right_knee"]
        )

    # Vidutiniai kampai iš galiojančių reikšmių
    angles["average_knee_angle"] = average_valid_angles(
        angles["left_knee_angle"],
        angles["right_knee_angle"]
    )

    angles["average_hip_angle"] = average_valid_angles(
        angles["left_hip_angle"],
        angles["right_hip_angle"]
    )

    return angles


def average_valid_angles(angle1, angle2):
    """
    Apskaičiuoja dviejų kampų vidurkį.
    Jei vienas kampas negaliojantis, grąžina kitą.
    """

    valid_angles = []

    if angle1 is not None:
        valid_angles.append(angle1)

    if angle2 is not None:
        valid_angles.append(angle2)

    if len(valid_angles) == 0:
        return None

    return sum(valid_angles) / len(valid_angles)


def evaluate_squat(angles):
    """
    Įvertina pritūpimą pagal apskaičiuotus kampus.

    Čia naudojamos paprastos taisyklės:
    - jei kelio kampas mažesnis nei 100 laipsnių, laikoma, kad pritūpimas pakankamai gilus;
    - jei kelio kampas didelis, žmogus dar nėra pakankamai pritūpęs.
    """

    knee_angle = angles["average_knee_angle"]
    hip_angle = angles["average_hip_angle"]

    evaluation = {
        "is_squat_deep_enough": False,
        "feedback": []
    }

    if knee_angle is None:
        evaluation["feedback"].append("Nepavyko apskaičiuoti kelio kampo.")
        return evaluation

    if knee_angle < 100:
        evaluation["is_squat_deep_enough"] = True
        evaluation["feedback"].append("Pritūpimo gylis pakankamas.")
    else:
        evaluation["feedback"].append("Pritūpimas per negilus.")

    if hip_angle is not None:
        if hip_angle < 60:
            evaluation["feedback"].append("Klubų kampas rodo gilų pritūpimą.")
        else:
            evaluation["feedback"].append("Klubai dar nepakankamai nusileidę.")

    return evaluation

class SquatTracker:
    """
    Sekamas visas pritūpimo judesys.
    Įvertinimas pateikiamas tik tada, kai pritūpimas užbaigiamas.
    """

    def __init__(self):
        self.state = "standing"

        self.min_knee_angle = None
        self.min_hip_angle = None

        self.last_feedback = None
        self.squat_count = 0

        # Ribos, kurias galima koreguoti testavimo metu
        self.start_squat_knee_angle = 150   # kai kampas mažesnis, laikoma, kad pradėjo tūpti
        self.deep_squat_knee_angle = 100    # jei pasiekia mažiau nei 100, gylis geras
        self.standing_knee_angle = 160      # kai vėl viršija, laikoma, kad atsistojo

    def update(self, angles):
        """
        Šią funkciją reikia kviesti kiekviename kadre.
        Ji grąžina feedback tik tada, kai pritūpimas užbaigiamas.
        """

        if angles is None:
            return None

        knee_angle = angles["average_knee_angle"]
        hip_angle = angles["average_hip_angle"]

        if knee_angle is None:
            return None

        # 1 būsena: žmogus stovi
        if self.state == "standing":
            if knee_angle < self.start_squat_knee_angle:
                self.state = "squatting_down"
                self.min_knee_angle = knee_angle
                self.min_hip_angle = hip_angle

            return None

        # 2 būsena: žmogus jau daro pritūpimą
        elif self.state == "squatting_down":
            # Saugome mažiausią kelio kampą per visą pritūpimą
            if self.min_knee_angle is None or knee_angle < self.min_knee_angle:
                self.min_knee_angle = knee_angle

            if hip_angle is not None:
                if self.min_hip_angle is None or hip_angle < self.min_hip_angle:
                    self.min_hip_angle = hip_angle

            # Jei žmogus vėl atsistojo, pritūpimas baigtas
            if knee_angle > self.standing_knee_angle:
                self.state = "standing"
                self.squat_count += 1

                feedback = self.evaluate_completed_squat()

                self.last_feedback = feedback

                # Paruošiame kitam pritūpimui
                self.min_knee_angle = None
                self.min_hip_angle = None

                return feedback

            return None

    def evaluate_completed_squat(self):
        """
        Įvertina jau užbaigtą pritūpimą.
        """

        feedback = {
            "squat_count": self.squat_count,
            "min_knee_angle": self.min_knee_angle,
            "min_hip_angle": self.min_hip_angle,
            "is_good_squat": False,
            "message": ""
        }

        if self.min_knee_angle is None:
            feedback["message"] = "Pritūpimo įvertinti nepavyko."
            return feedback

        if self.min_knee_angle < self.deep_squat_knee_angle:
            feedback["is_good_squat"] = True
            feedback["message"] = "Pritūpimas atliktas tinkamai."
        else:
            feedback["is_good_squat"] = False
            feedback["message"] = "Pritūpimas per negilus."

        return feedback