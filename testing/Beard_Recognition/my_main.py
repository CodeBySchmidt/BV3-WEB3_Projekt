import numpy as np
import cv2
import dlib
import os

from typing import Tuple, List, Dict, Union
from onnxruntime import InferenceSession


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


# OWN FACEDETCTOR -> USED IN LANDMARKS FOR EVERY DETECTOR
class FaceDetector:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


# REWORKED CODE TO FIT IN LANDMARKS
class BeardDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def process_image(self, image_path: str, use_detector: bool, draw_landmarks: bool):
        """
        Process a single image and return the prediction of the beard (and the Image (if drawing landmarks == True)).

        :param image_path: Path to the image file.
        :param use_detector: If False, classify the whole image without face detection.

        My own parameter:
        :param draw_landmarks: If True, display landmarks.
        """

        COORDINATES_EXTEND_VALUE = 0.2

        INPUT_SHAPE = (128, 128, 3)
        CLASSIFIER_MODEL_PATH = "../Beard_Recognition/hat_beard_classifier/hat_beard_model.onnx"

        # Debugging information.
        if not os.path.exists(CLASSIFIER_MODEL_PATH):
            print(f"Model file does not exist at path: {CLASSIFIER_MODEL_PATH}")
        else:
            print(f"Model file found at path: {CLASSIFIER_MODEL_PATH}")

        classifier = HatBeardClassifier(CLASSIFIER_MODEL_PATH, INPUT_SHAPE)

        image = cv2.imread(image_path)

        if image is None:
            print(f'Kann das Bild nicht lesen: "{image_path}".')
            return

        if use_detector:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                print("Keine Gesichter erkannt.")
                return

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                x, y, w, h = get_coordinates(image, (x, y, w, h), COORDINATES_EXTEND_VALUE)
                class_result = classifier.inference(image[y:y + h, x:x + w, :])
                beard_value = class_result['beard']

                if draw_landmarks:
                    # Vorhersage der Gesichtsmerkmale (Landmarks)
                    landmarks = self.predictor(gray, face)
                    # Zeichne die Landmarks auf das Bild
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

                return beard_value, image

        else:
            class_result = classifier.inference(image)
            beard_value = class_result['beard']
            return beard_value, None


if __name__ == '__main__':

    # Path to Image
    image_path = "1.jpg"

    PREDICTOR_PATH = "../Beard_Recognition/hat_beard_classifier/shape_predictor_68_face_landmarks.dat"


    # Initialisierung des Detektors und Klassifikators
    beard_detector = BeardDetector(PREDICTOR_PATH)

    # Verarbeitung des Bildes
    # Bestimmen, ob der Gesichtserkennungsdetektor verwendet werden soll
    use_detector = True  # Ändere zu False, um den Detektor nicht zu verwenden
    draw_landmarks = True  # Ändere zu False, um die Landmarks nicht zeichnen zu lassen
    predicted_output, result_image = beard_detector.process_image(image_path, use_detector, draw_landmarks)

    # Optional Code for Checking the predicted_output -> Is not needed in Landmarks
    print(f"Prediction for the Image {image_path}: Beard = {predicted_output * 100} %")
    image = cv2.imread(image_path)
    if draw_landmarks:
        cv2.imshow("Result Image", result_image)
        cv2.waitKey(0)
    else:
        cv2.imshow("Input Image", image)
        cv2.waitKey(0)
