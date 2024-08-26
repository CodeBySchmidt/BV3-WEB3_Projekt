import os
import numpy as np
import dlib
import cv2
import webcolors
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import models
from PIL import Image
from imutils import face_utils

from typing import Tuple, List, Dict, Union
from onnxruntime import InferenceSession


class FaceDetector:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


class GlassesDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def detect_nose_region(self, img, rect):
        sp = self.predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28, 29, 30, 31, 33, 34, 35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])

        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]

        img_cropped = img[y_min:y_max, x_min:x_max]
        return img_cropped

    def detect_glasses(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist"

        img = dlib.load_rgb_image(image_path)

        faces = self.detect_face(img)

        if faces is None:
            return "No face detected"

        else:
            face = faces[0]
            img_cropped = self.detect_nose_region(img, face)
            img_blur = cv2.GaussianBlur(img_cropped, (9, 9), sigmaX=0, sigmaY=0)
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
            edges_center = edges.T[(int(len(edges.T) / 2))]

            if 255 in edges_center:
                return "Yes"
            else:
                return "No"

    def detect_face(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            return faces
        else:
            return None


class ColorFinder:
    def find_color(self, requested_colour):
        min_colours = {}
        for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(name)
            bd = (b_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            rd = (r_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = key
        closest_name = min_colours[min(min_colours.keys())]
        return closest_name


def calculate_median_color(image):
    """
    Berechnet die Medianfarbe eines Bildes und gibt sowohl die RGB-Werte
    als auch den Hex-Farbwert zurück.

    Args:
        image (np.array): Das Bild als numpy-Array.

    Returns:
        tuple: Die Medianfarbe als ((B, G, R), Hex-Farbwert)-Tupel, falls das Bild farbig ist,
               oder als ((Grauwert, Grauwert, Grauwert), Hex-Farbwert)-Tupel, falls das Bild
               ein Graustufenbild ist.
    """
    if image is None:
        return "No Face detected", "No Face detected"

    if len(image.shape) == 3:
        median_r = np.median(image[:, :, 0].flatten())
        median_g = np.median(image[:, :, 1].flatten())
        median_b = np.median(image[:, :, 2].flatten())
        median_color = (median_r, median_g, median_b)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(median_r), int(median_g), int(median_b))
        return median_color, hex_color
    else:
        median_value = np.median(image.flatten())
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(median_value), int(median_value), int(median_value))
        return (median_value, median_value, median_value), hex_color


class HairColorDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            return faces
        else:
            return None

    def find_hair_color(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist", "Image does not exist"

        image = dlib.load_rgb_image(image_path)
        faces = self.detect_face(image)

        if faces is None:
            return "No face detected", "No face detected"

        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = faces[0]
            landmarks = self.predictor(gray, face)
            face_utils.shape_to_np(landmarks)
            (x, y, w, h) = face_utils.rect_to_bb(face)

            # Berechnung der Bounding Box
            x1 = max(0, x)
            y1 = max(0, y - h // 2)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

            cropped_image = image[y1:y2, x1:x2]
            img = image[y:y + h, x:x + w]
            # Oberer Teil des Gesichts
            upper_part = cropped_image[:int(cropped_image.shape[0] * 0.25), :]

            median_color_hair, hair_hex = calculate_median_color(upper_part)
            median_color_skin, skin_hex = calculate_median_color(img)

            # color_finder = ColorFinder()
            # color_name = color_finder.find_color(median_color_hair)

            # Vergleich der Medianfarben
            hair_diff = np.sqrt((median_color_hair[0] - median_color_skin[0]) ** 2 +
                                (median_color_hair[1] - median_color_skin[1]) ** 2 +
                                (median_color_hair[2] - median_color_skin[2]) ** 2)

            # Schwellwert für den Farbunterschied
            threshold = 60

            if hair_diff < threshold:
                return hair_hex, "Probably a bald spot or a bald head"
            else:
                return hair_hex, "Possible not bald or balding"


class EyeColorDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)
        self.flag = 0

    def find_color(self, requested_colour):
        min_colours = {}
        for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(name)
            bd = (b_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            rd = (r_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = key
        closest_name = min_colours[min(min_colours.keys())]
        return closest_name

    def detect_eye_color(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist"

        faces, gray, img_rgb = self.detect_face(image_path)
        if faces is None:
            return "No face detected"

        for face in faces:
            eyes = []  # Liste zur Speicherung der Augen

            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)

            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

            left_eye = shape[left_eye_start:left_eye_end]
            right_eye = shape[right_eye_start:right_eye_end]

            eyes.append(left_eye)
            eyes.append(right_eye)

            for index, eye in enumerate(eyes):
                self.flag += 1
                left_side_eye = eye[0]
                right_side_eye = eye[3]
                top_side_eye = eye[1]
                bottom_side_eye = eye[4]

                eye_width = right_side_eye[0] - left_side_eye[0]
                eye_height = bottom_side_eye[1] - top_side_eye[1]

                eye_x1 = int(left_side_eye[0] - 0 * eye_width)
                eye_x2 = int(right_side_eye[0] + 0 * eye_width)
                eye_y1 = int(top_side_eye[1] - 1 * eye_height)
                eye_y2 = int(bottom_side_eye[1] + 0.75 * eye_height)

                roi_eye = img_rgb[eye_y1:eye_y2, eye_x1:eye_x2]

                if self.flag == 1:
                    break

        if roi_eye.size == 0:
            return "Eye region not found"

        x = roi_eye.shape
        row = x[0]
        col = x[1]

        # Definiere die Schwellwerte
        upper_threshold = 230
        lower_threshold = 30

        # Definiere die ROI für die Pixel direkt unter der Pupille
        roi_px = roi_eye[(row // 2):(row // 2) + 15, int((col // 3) + 10):int((col // 3)) + 25].reshape(-1, 3)

        # Filtere Pixel basierend auf den Schwellwerten
        filtered_array1 = np.array(
            [pixel for pixel in roi_px if all(lower_threshold <= channel <= upper_threshold for channel in pixel)])

        if len(filtered_array1) > 0:
            array1 = np.median(filtered_array1, axis=0).astype(int)
        else:
            array1 = [0, 0, 0]

        array1 = tuple(array1)

        color_name = self.find_color(array1)

        return color_name

    def detect_face(self, image):
        image = cv2.imread(image)
        if image is None:
            return None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            return faces, gray, image
        else:
            return None, gray, image


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(in_features, 18)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        age, gender, race = torch.split(x, [9, 2, 7], dim=1)
        return age, gender, race


class AgeGenderRaceDetector(FaceDetector):

    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            return faces
        else:
            return None

    def predict(self, image_path):

        if not os.path.isfile(image_path):
            return "Image does not exist"

        image_loaded = dlib.load_rgb_image(image_path)

        detected_face = self.detect_face(image_loaded)

        if detected_face is None:
            return "No face detected", "No face detected", "No face detected"

        else:
            # image_path = self.image_path
            # Define the same transformations as used during training
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # Load and transform the image
            image = Image.open(image_path)
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

            # Load the model
            model = ConvNet()
            model.load_state_dict(torch.load(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'best_fairface_model2.pth')),
                map_location=torch.device('cpu')))
            model.eval()

            # Make predictions
            with torch.no_grad():
                pred_age, pred_gender, pred_race = model(image)

            # Get the predicted labels
            pred_age = torch.argmax(pred_age, dim=1).item()
            pred_gender = torch.argmax(pred_gender, dim=1).item()
            pred_race = torch.argmax(pred_race, dim=1).item()

            # Mapping dictionaries
            age_mapping = {
                0: '0-2',
                1: '3-9',
                2: '10-19',
                3: '20-29',
                4: '30-39',
                5: '40-49',
                6: '50-59',
                7: '60-69',
                8: 'more than 70'
            }

            gender_mapping = {
                0: 'Male',
                1: 'Female'
            }

            race_mapping = {
                0: 'White',
                1: 'Black',
                2: 'Latino_Hispanic',
                3: 'East Asian',
                4: 'Southeast Asian',
                5: 'Indian',
                6: 'Middle Eastern'
            }

            # Convert predictions to labels
            pred_age_label = age_mapping[pred_age]
            pred_gender_label = gender_mapping[pred_gender]
            pred_race_label = race_mapping[pred_race]
            return pred_age_label, pred_gender_label, pred_race_label


# ORIGINAL GIT CODE
class OnnxModelLoader:
    def __init__(self, onnx_path: str) -> None:
        """
        Class for loading ONNX models to inference on CPU. CPU inference is very effective using onnxruntime.

        :param onnx_path: path to ONNX model file (*.onnx file).
        """
        self.sess = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        self.input_name = [x.name for x in self.sess.get_inputs()][0]
        self.output_names = [x.name for x in self.sess.get_outputs()]

    def inference(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        Run inference.

        :param inputs: list of arguments, order must match names in input_names.
        :return: list of outputs.
        """
        return self.sess.run(self.output_names, input_feed={self.input_name: inputs})


# ORIGINAL GIT CODE
class HatBeardClassifier:
    def __init__(self, model_path: str, input_shape: Tuple[int, int, int]) -> None:
        """
        Class for easy using of hat/beard classifier.

        :param model_path: path to trained model, converted to ONNX format.
        :param input_shape: input shape tuple (height, width, channels).
        """
        self.input_shape = input_shape

        self.model = OnnxModelLoader(model_path)
        self.class_names = ('No hat, no beard', 'Hat', 'Beard', 'Hat and beard')

    def inference(self, image: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Process image and return class name with probabilities for presence of hat and beard on the image.
        Example of returning dict:
        {
            'class': 'No hat, no beard',
            'hat': 0.05,
            'beard': 0.01
        }

        :param image: image in BGR format (obtained using cv2) to process.
        :return: dict with results.
        """
        img = preprocess_image(image, self.input_shape)
        predictions = self.model.inference(img)
        class_name, hat_prob, beard_prob = self._get_class(predictions)
        return {'class': class_name, 'hat': hat_prob, 'beard': beard_prob}

    def _get_class(self, predictions: List[np.ndarray]) -> Tuple[str, float, float]:
        """
        Get predicted class name and probabilities for each class.

        :param predictions: list of two predicted arrays (hat one-hot and beard one-hot).
        :return: class name and probabilities for each class.
        """
        hat_labels = predictions[0][0]
        beard_labels = predictions[1][0]

        hat_label = int(np.argmax(hat_labels))
        beard_label = int(np.argmax(beard_labels))
        if hat_label == 1 and beard_label == 1:
            return self.class_names[0], hat_labels[0], beard_labels[0]
        elif hat_label == 0 and beard_label == 1:
            return self.class_names[1], hat_labels[0], beard_labels[0]
        elif hat_label == 1 and beard_label == 0:
            return self.class_names[2], hat_labels[0], beard_labels[0]
        else:
            return self.class_names[3], hat_labels[0], beard_labels[0]


# ORIGINAL GIT CODE
def preprocess_image(image: np.ndarray, input_shape: Tuple[int, int, int], bgr_to_rgb: bool = True) -> np.ndarray:
    """
    Copy input image and preprocess it for further inference.

    :param image: image numpy array in RGB or BGR format.
    :param input_shape: input shape tuple (height, width, channels).
    :param bgr_to_rgb: if True, then convert image from BGR to RGB.
    :return: image array ready for inference.
    """
    img = image.copy()
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape[:2][::-1], interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img / 255.0, axis=0)
    return np.float32(img)


# ORIGINAL GIT CODE
def get_coordinates(image: np.ndarray, coordinates: List[int], extend_value: float) -> Tuple[int, int, int, int]:
    """
    Get extended coordinates of found face for accurate hat/beard classification.

    :param image: original image.
    :param coordinates: found face coordinates in format [x, y, w, h].
    :param extend_value: positive float < 1.
    :return: obtained coordinates in same format.
    """
    x, y, w, h = coordinates
    x = int(np.clip(x - extend_value * w, 0, image.shape[1]))
    y = int(np.clip(y - extend_value * h, 0, image.shape[0]))
    w = int(np.clip(w * (1 + extend_value), 0, image.shape[1]))
    h = int(np.clip(h * (1 + extend_value), 0, image.shape[0]))
    return x, y, w, h


# REWORKED CODE TO FIT IN LANDMARKS
class BeardDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) > 0:
            return faces
        else:
            return None

    def process_image(self, image_path: str):
        """
        Process a single image and return the prediction of the beard (and the Image (if drawing landmarks == True)).
        :param image_path: Path to the image file.
        """

        if not os.path.isfile(image_path):
            return "Image does not exist"

        image_loaded = dlib.load_rgb_image(image_path)
        detected_face = self.detect_face(image_loaded)

        if detected_face is None:
            return "No face detected", "No face detected", "No face detected"
        else:
            COORDINATES_EXTEND_VALUE = 0.2
            INPUT_SHAPE = (128, 128, 3)
            CLASSIFIER_MODEL_PATH = "../Utils/hat_beard_model.onnx"

            # Debugging information.
            if not os.path.exists(CLASSIFIER_MODEL_PATH):
                print(f"Model file does not exist at path: {CLASSIFIER_MODEL_PATH}")
            else:
                print(f"Model file found at path: {CLASSIFIER_MODEL_PATH}")

            classifier = HatBeardClassifier(CLASSIFIER_MODEL_PATH, INPUT_SHAPE)

            image = cv2.imread(image_path)

            for face in detected_face:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                x, y, w, h = get_coordinates(image, (x, y, w, h), COORDINATES_EXTEND_VALUE)
                class_result = classifier.inference(image[y:y + h, x:x + w, :])
                beard_value = class_result['beard']
                return beard_value


# Alter Code

# class FaceLandmarkDetector:
#     def __init__(self, predictor_path, camera_index=0):
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = dlib.shape_predictor(predictor_path)
#         self.cap = cv2.VideoCapture(camera_index)
#
#         if not self.cap.isOpened():
#             raise RuntimeError(
#                 "Kamera konnte nicht geöffnet werden. Überprüfen Sie den Kamera-Index und ob die Kamera angeschlossen "
#                 "ist.")
#
#     def detect_and_draw_landmarks(self, image, draw_circles=True):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         rects = self.detector(gray, 0)
#
#         for (i, rect) in enumerate(rects):
#             shape = self.predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#
#             # Nur Kreise zeichnen, wenn draw_circles True ist
#             if draw_circles:
#                 for (x, y) in shape:
#                     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#
#         return image
#
#     def get_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             return None
#         frame_with_circles = self.detect_and_draw_landmarks(frame)
#         return cv2.cvtColor(frame_with_circles, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter
#
#     def save_screenshot(self, file_path):
#         ret, frame = self.cap.read()
#         if ret:
#             frame_without_circles = self.detect_and_draw_landmarks(frame, draw_circles=False)
#             cv2.imwrite(file_path, frame_without_circles)
#             print(f"Screenshot saved to {file_path}")
#             return frame_without_circles
#         else:
#             print("Failed to capture image.")
#
#     # take_screenshot braucht man eventuell nicht mehr
#     def take_screenshot(self, file_path):
#         time.sleep(5)  # Wait for 5 seconds
#         self.save_screenshot(file_path)
#
#     def show_live_video(self):
#         screenshot_path = "screenshot.jpg"
#         screenshot_thread = threading.Thread(target=self.save_screenshot, args=(screenshot_path,))
#         screenshot_thread.start()
#
#         start_time = time.time()
#         while True:
#             ret, frame = self.cap.read()
#             if not ret or time.time() - start_time > 6:
#                 break
#             frame_with_circles = self.detect_and_draw_landmarks(frame)
#             cv2.imshow("Live Video - Press ESC to exit", frame_with_circles)
#
#             k = cv2.waitKey(5) & 0xFF
#             if k == 27:  # ESC key to break
#                 break
#
#         screenshot_thread.join()  # Warten Sie darauf, dass der Screenshot-Thread abgeschlossen ist
#         self.cap.release()
#         cv2.destroyAllWindows()
#
#     def update_image(self, img_element):
#         while True:
#             frame = self.get_frame()
#             if frame is None:
#                 break
#             # Convert the frame to a format suitable for display in niceGUI
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue
#             img_element.source = f'data:image/jpeg;base64,{buffer.tobytes().encode("base64").decode()}'
#             time.sleep(0.1)


# if __name__ == "__main__":
#     try:
#         predictor_path = "../Utils/shape_predictor_68_face_landmarks.dat"
#         selected_camera_index = int(input("Geben Sie den Index der Kamera ein, die verwendet werden soll (0 oder 1): "))
#
#         detector = FaceDetector(predictor_path)
#         glasses_detector = GlassesDetector(predictor_path)
#         eye_color_detector = EyeColorDetector(predictor_path)
#         face_landmark_detector = FaceLandmarkDetector(predictor_path, camera_index=selected_camera_index)
#
#     # Starte das Live Video
#         face_landmark_detector.show_live_video()
#         print()
#         time.sleep(2)
#
#     # Brillen - Erkennung wird ausgeführt
#         seconds = 3
#         print(f"Die Methode {"glasses_detector.display_results()"} wird aufgerufen...")
#         time.sleep(2)
#         # Schleife für die Ausgabe
#         for i in range(seconds):
#             print(f"Berechnung dauert noch {seconds - i} Sekunden....")
#             time.sleep(1)
#         print(f"Ergebnis ist: " + glasses_detector.display_results("screenshot.jpg"))
#         print()
#         time.sleep(2)
#
#     # Augenfarben - Erkennung wird ausgeführt
#         seconds = 3
#         print(f"Die Methode {"eye_color_detector.detect_eye_color()"} wird aufgerufen...")
#         time.sleep(2)
#         # Schleife für die Ausgabe
#         for i in range(seconds):
#             print(f"Berechnung dauert noch {seconds - i} Sekunden...")
#             time.sleep(1)
#         print(f"Ergebnis ist: " + eye_color_detector.detect_eye_color("screenshot.jpg"))
#
#     except Exception as e:
#         print(f"Fehler beim Ausführen: {e}")

