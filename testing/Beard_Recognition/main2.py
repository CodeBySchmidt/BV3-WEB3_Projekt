import os
import argparse
from typing import Union
import cv2
from typing import List, Tuple, Dict, Union
import dlib
import numpy as np
from onnxruntime import InferenceSession

# Hat/beard classifier parameters.
INPUT_SHAPE = (128, 128, 3)
CLASSIFIER_MODEL_PATH = os.path.join('testing', 'Beard_Recognition', 'hat_beard_classifier', 'hat_beard_model.onnx')

# Debugging information.
if not os.path.exists(CLASSIFIER_MODEL_PATH):
    print(f"Model file does not exist at path: {CLASSIFIER_MODEL_PATH}")
else:
    print(f"Model file found at path: {CLASSIFIER_MODEL_PATH}")

# Increase the size of the found face region by this fraction.
# It is necessary for further accurate hat/beard classification.
COORDINATES_EXTEND_VALUE = 0.2


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


def draw_results(image: np.ndarray, faces: List[List[int]],
                 hats_beards: List[Dict[str, Union[str, float]]]) -> np.ndarray:
    """
    Draw found face and predicted image class on original image.

    :param image: original image.
    :param faces: list with faces coordinates.
    :param hats_beards: list with classification results for each face respectively.
    :return: image with bounding boxes.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 0, 255)
    for (x, y, w, h), hat_beard_dict in zip(faces, hats_beards):
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = '{}. hat = {:.02f}%, beard = {:.02f}%'.format(
            hat_beard_dict['class'], hat_beard_dict['hat'] * 100, hat_beard_dict['beard'] * 100
        )
        (label_width, label_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        if x + label_width > image.shape[1]:
            _image = np.zeros((image.shape[0], x + label_width, image.shape[2]), dtype=np.uint8)
            _image[:, :image.shape[1], :] = image
            image = _image
        cv2.rectangle(image, (x, y - label_height - baseline), (x + label_width, y), color, -1)
        cv2.putText(image, text, (x, y - baseline // 2), font,
                    font_scale, (0, 0, 0), lineType=cv2.LINE_AA, thickness=thickness)
    return image


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

class SimpleFaceDetector:
    def __init__(self, scale_factor: float = 1.0) -> None:
        """
        Wrapper for face detection using dlib.

        :param scale_factor: scale_factor for image resizing, float > 1.
        """
        self.scale_factor = scale_factor
        self.detector = dlib.get_frontal_face_detector()

    def inference(self, image: np.ndarray) -> List[List[int]]:
        """
        Detect faces in image using dlib.

        :param image: image in BGR format (obtained using cv2).
        :return: list of found faces (each one is [x, y, w, h] - bounding box coordinates).
        """
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        detections = self.detector(resized_img)
        faces = [[d.left(), d.top(), d.width(), d.height()] for d in detections]
        return faces
    
def process_image(image_path: str, use_detector: bool) -> None:
    """
    Process a single image and show it. Press any key to continue, press "q" to exit.

    :param image_path: path to the image file.
    :param use_detector: if False, then don't use face detector and classify whole image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print('Can\'t read image: "{}".'.format(image_path))
        return
    if use_detector:
        faces = detector.inference(image)
        classes = []
        for face_coordinates in faces:
            x, y, w, h = get_coordinates(image, face_coordinates, COORDINATES_EXTEND_VALUE)
            class_result = classifier.inference(image[y:y + h, x:x + w, :])
            classes.append(class_result)
            print(f'Beard probability for detected face: {class_result["beard"]}')
        image = draw_results(image, faces, classes)
    else:
        class_result = classifier.inference(image)
        print(f'Beard probability for the whole image: {class_result["beard"]}')
        image = draw_results(image, [[0, image.shape[0] - 1, 0, 0]], [class_result])
    cv2.imshow('Image', image)
    if cv2.waitKey(0) == ord('q'):
        return class_result["beard"]


def parse_args() -> argparse.Namespace:
    """
    Parsing command line arguments with argparse.
    """
    parser = argparse.ArgumentParser(
        'Script for processing images or video to find faces and find headdress and beard on it.'
    )
    parser.add_argument('--no_detector', action='store_true', help='Don\'t use face detector while processing the image.')
    return parser.parse_args()

if __name__ == '__main__':
    SCALE_FACTOR = 1.0  # Set the scale factor as needed
    detector = SimpleFaceDetector(SCALE_FACTOR)
    classifier = HatBeardClassifier(CLASSIFIER_MODEL_PATH, INPUT_SHAPE)

    IMAGE_PATH = ('E:\BV3&WEB3 Projekt\current_frame.jpg')
    args = parse_args()
    process_image(IMAGE_PATH, not args.no_detector)
