from video_landmarks import EyeColorDetector, GlassesDetector, AgeGenderRaceDetector, HairColorDetector, BeardDetector
import os


image_path = "current_frame.jpg"  # Hier den Pfad zu deinem Bild einfügen

age_proto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'age_deploy.prototxt'))
age_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'age_net.caffemodel'))
gender_proto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'gender_deploy.prototxt'))
gender_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'gender_net.caffemodel'))
predictor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))


async def eye_color() -> str:
    try:
        eye_color_detector = EyeColorDetector(predictor_path)
        # result_eye_color = eye_color_detector.detect_eye_color(image_path)
        result_eye_color = "EyeColorDetector"
        return result_eye_color
    except Exception as e:
        print(f"Error in eye_color function: {e}")


async def glasses() -> str:
    try:
        glasses_detector = GlassesDetector(predictor_path)
        result_glasses = glasses_detector.detect_glasses(image_path)
        # result_glasses = "GlassesDetector"
        return result_glasses

    except Exception as e:
        print(f"Error in glasses function: {e}")


async def facial_hair() -> str:
    try:
        beard_detector = BeardDetector(predictor_path)
        result_beard = beard_detector.process_image(image_path)
        return result_beard
    except Exception as e:
        print(f"Error in facial_hair function: {e}")
    # return "None"


async def hair_color():
    try:
        hair_type_detector = HairColorDetector(predictor_path)
        result_hair_color, result_hair_typ = hair_type_detector.find_hair_color(image_path)
        return result_hair_color, result_hair_typ
    except Exception as e:
        print(f"Error in hair_color function: {e}")
    # return "None"


async def gender_age_race() -> str:
    # detector = GenderAgeDetector(age_model, age_proto, gender_model, gender_proto, predictor_path)
    # gender_result = detector.process_image(image_path)[0]
    # age_result = detector.process_image(image_path)[1]
    # return gender_result, age_result
    try:
        detector = AgeGenderRaceDetector(predictor_path)
        age_result, gender_result, race_result = detector.predict(image_path)
        return age_result, gender_result, race_result
    except Exception as e:
        print(f"Error in gender_age_race function: {e}")
