import cv2
import numpy as np
from imutils import face_utils
import dlib
import matplotlib.pyplot as plt

class FaceDetector:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

class HairColorDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def crop_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        for (i, rect) in enumerate(rects):
            landmarks = self.predictor(gray, rect)
            face_utils.shape_to_np(landmarks)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            x1 = x
            y1 = y - int(h / 2)
            x2 = x + w
            y2 = y + h

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cropped_image = image[y1:y2, x1:x2]
            img = image[y:y + w, x:x + h]

            height, width = cropped_image.shape[:2]
            upper_part_height = int(height * 0.25)
            upper_part = cropped_image[:upper_part_height, :]

            return upper_part, img

    def find_hair_color(self, image_path):
        image = cv2.imread(image_path)
        upper_part, img = self.crop_face(image)

        # Haarfarbe
        if len(upper_part.shape) == 3:
            median_b_h = np.median(upper_part[:, :, 0].flatten())
            median_g_h = np.median(upper_part[:, :, 1].flatten())
            median_r_h = np.median(upper_part[:, :, 2].flatten())
            median_color_hair = (median_r_h, median_g_h, median_b_h)
            hex_color_hair = '#{:02x}{:02x}{:02x}'.format(int(median_r_h), int(median_g_h), int(median_b_h))
        else:
            median_value = np.median(upper_part.flatten())
            median_color_hair = (median_value, median_value, median_value)

        # Hautfarbe
        if len(img.shape) == 3:
            median_b = int(np.median(image[:, :, 0].flatten()))
            median_g = int(np.median(image[:, :, 1].flatten()))
            median_r = int(np.median(image[:, :, 2].flatten()))
            median_color_skin = (median_b, median_g, median_r)
        else:
            median_value = np.median(img.flatten())
            median_color_skin = (median_value, median_value, median_value)

        # Vergleich der Medianfarben
        hair_diff = np.sqrt((median_color_hair[0] - median_color_skin[0]) ** 2 +
                            (median_color_hair[1] - median_color_skin[1]) ** 2 +
                            (median_color_hair[2] - median_color_skin[2]) ** 2)
        # print(f"Unterschied der Farbwerte von Haar und Haut: {hair_diff}")

        # Schwellwert für den Farbunterschied
        threshold = 60

        # Ein neues Bild mit dem Median-Farbwert erstellen
        hair_color_median = np.full((300, 300, 3), median_color_hair, dtype=np.uint8)

        # Ein neues Bild mit dem Mittelwert-Farbwert erstellen
        # hair_color_hex = np.full((300, 300, 3), hex_color_hair, dtype=np.uint8)

        print(median_color_hair)
        print(hex_color_hair)

        # Plotten der Bilder
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title("Median")
        plt.imshow(hair_color_median)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Hex')
        plt.figtext(0.5, 0.01, f'Hair Color Hex: {hex_color_hair}', ha='center', fontsize=12, color=hex_color_hair, bbox=dict(facecolor='white', alpha=0.7))
        plt.axis('off')

        plt.show()

        if hair_diff < threshold:
            return "Glatze"
        else:
            return "Keine Glatze"

# Nutzung der Klasse
image_path = "Utils/img/2.jpg"
landmark_detector_path = "Utils/shape_predictor_68_face_landmarks.dat"
processor = HairColorDetector(landmark_detector_path)
result = processor.find_hair_color(image_path)

print(result)
