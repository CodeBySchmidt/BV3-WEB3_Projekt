# Quelle: https://medium.com/@siddh30/glasses-detection-opencv-dlib-bf4cd50856da

import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class GlassesDetector:
    def __init__(self, predictor_path):
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_nose_region(self, image_path):
        img = dlib.load_rgb_image(image_path)
        rect = dlib.get_frontal_face_detector()(img)[0]
        sp = self.predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])

        nose_bridge_x = []
        nose_bridge_y = []
        for i in [28, 29, 30, 31, 33, 34, 35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])

        # x_min and x_max
        x_min = min(nose_bridge_x)
        x_max = max(nose_bridge_x)

        # ymin (from top eyebrow coordinate),  ymax
        y_min = landmarks[20][1]
        y_max = landmarks[31][1]

        img_cropped = img[y_min:y_max, x_min:x_max]
        plt.imshow(img)
        plt.title("Input Image")
        plt.show()
        plt.imshow(img_cropped)
        plt.title("Cropped Image")
        plt.show()

        return x_min, x_max, y_min, y_max

    def detect_glasses(self, image_path):
        x_min, x_max, y_min, y_max = self.detect_nose_region(image_path)
        img = Image.open(image_path).crop((x_min, y_min, x_max, y_max))
        img_blur = cv2.GaussianBlur(np.array(img), (9, 9), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        plt.imshow(img_blur)
        plt.title("Blur Image")
        plt.show()

        plt.imshow(edges, cmap='gray')
        plt.title("Edges Image")
        plt.show()

        edges_center = edges.T[(int(len(edges.T) / 2))]

        # print(edges_center)

        if 255 in edges_center:
            return True
        else:
            return False

    def display_results(self, img_path):
        glasses = self.detect_glasses(img_path)
        if glasses:
            print('Trägt Brille')
        else:
            print('Keine Brille')


if __name__ == "__main__":

    predictor_path = "Utils/shape_predictor_68_face_landmarks.dat"
    glasses_detector = GlassesDetector(predictor_path)

    image_path = "img/glasses/1.jpg"
    glasses_detector.display_results(image_path)
