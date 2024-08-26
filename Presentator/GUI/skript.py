from video_landmarks import FaceDetector, EyeColorDetector, GlassesDetector, AgeGenderRaceDetector, HairColorDetector
import os


image_path = "current_frame.jpg"  # Hier den Pfad zu deinem Bild einfügen

age_proto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'age_deploy.prototxt'))
age_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'age_net.caffemodel'))
gender_proto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'gender_deploy.prototxt'))
gender_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'gender_net.caffemodel'))
predictor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))

# Wird eventuell gar nicht benötigt
# def get_screenshot():
#     detector = FaceLandmarkDetector(predictor_path)
#     image = detector.take_screenshot("screenshot.jpg")
#     return image


async def eye_color() -> str:
    eye_color_detector = EyeColorDetector(predictor_path)
    result_eye_color_hex, eye_color_name = eye_color_detector.detect_eye_color(image_path)
    return result_eye_color_hex, eye_color_name


async def glasses() -> str:
    glasses_detector = GlassesDetector(predictor_path)
    result_glasses = glasses_detector.detect_glasses(image_path)
    return result_glasses


async def facial_hair() -> str:
    return "None"


async def hair_color():
    hair_type_detector = HairColorDetector(predictor_path)
    result_hair_hex, hair_color_name, result_hair_typ = hair_type_detector.find_hair_color(image_path)
    return result_hair_hex, hair_color_name, result_hair_typ
    # return "None"


async def gender_age_race() -> str:
    # detector = GenderAgeDetector(age_model, age_proto, gender_model, gender_proto, predictor_path)
    # gender_result = detector.process_image(image_path)[0]
    # age_result = detector.process_image(image_path)[1]
    # return gender_result, age_result
    detector = AgeGenderRaceDetector(predictor_path)
    age_result, gender_result, race_result = detector.predict(image_path)
    return age_result, gender_result, race_result
