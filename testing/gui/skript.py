from video_landmarks import EyeColorDetector, GlassesDetector, GenderAgeDetector


image_path = "current_frame.jpg"  # Hier den Pfad zu deinem Bild einfügen

age_proto = "../Utils/age_deploy.prototxt"
age_model = "../Utils/age_net.caffemodel"
gender_proto = "../Utils/gender_deploy.prototxt"
gender_model = "../Utils/gender_net.caffemodel"
predictor_path = "../Utils/shape_predictor_68_face_landmarks.dat"


# Wird eventuell gar nicht benötigt
# def get_screenshot():
#     detector = FaceLandmarkDetector(predictor_path)
#     image = detector.take_screenshot("screenshot.jpg")
#     return image


def eye_color() -> str:
    eye_color_detector = EyeColorDetector(predictor_path)
    result_eye_color = eye_color_detector.detect_eye_color(image_path)
    return result_eye_color


def glasses() -> str:
    glasses_detector = GlassesDetector(predictor_path)
    result_glasses = glasses_detector.display_results(image_path)
    return result_glasses


def hair_color() -> str:
    return "SaddleBrown"


def facial_hair() -> str:
    return "Yes"


def facial_hair_color() -> str:
    return "Fuchsia"


def gender_age() -> str:
    detector = GenderAgeDetector(age_model, age_proto, gender_model, gender_proto, predictor_path)
    gender_result = detector.process_image(image_path)[0]
    age_result = detector.process_image(image_path)[1]
    return gender_result, age_result


def race() -> str:
    return "European"

