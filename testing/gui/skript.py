import time

from video_landmarks import FaceLandmarkDetector, EyeColorDetector, FaceDetector, GlassesDetector 

predictor_path = "../Utils/shape_predictor_68_face_landmarks.dat"


def get_screenshot():
    detector = FaceLandmarkDetector(predictor_path)
    image = detector.take_screenshot("screenshot.jpg")
    return image


def eye_color() -> str:
    eye_color_detector = EyeColorDetector(predictor_path)
    result_eye_color = eye_color_detector.detect_eye_color("screenshot.jpg")
    return result_eye_color


def hair_color() -> str:
    return "SaddleBrown"


def facial_hair() -> str:
    return "Yes"


def facial_hair_color() -> str:
    return "Fuchsia"


def glasses() -> str:
    glasses_detector = GlassesDetector(predictor_path)
    result_glasses = glasses_detector.display_results("screenshot.jpg")
    return result_glasses


def age() -> str:
    return "25 - 30"


def race() -> str:
    return "European"


def gender() -> str:
    return "Female"

