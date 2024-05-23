import cv2
import dlib
from imutils import face_utils

class FaceLandmarkDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.cap = cv2.VideoCapture(0)

    def detect_and_draw_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        return image

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = self.detect_and_draw_landmarks(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter
