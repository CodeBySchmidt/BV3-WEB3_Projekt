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
            return "NO FACE DETECTED"

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
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
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
        return "NO FACE DETECTED", "NO FACE DETECTED"

    if len(image.shape) == 3:
        median_r = np.median(image[:, :, 0].flatten())
        median_g = np.median(image[:, :, 1].flatten())
        median_b = np.median(image[:, :, 2].flatten())
        median_color = (median_r, median_g, median_b)
        return median_color
    else:
        median_value = np.median(image.flatten())
        return median_value


def rgb_to_hex(rgb):
    """
    Convert RGB color to hexadecimal representation.

    Args:
        rgb (tuple): A tuple containing the RGB values, e.g., (255, 0, 0).

    Returns:
        str: The hexadecimal color code, e.g., '#ff0000'.
    """
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


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
            return "Image does not exist", "Image does not exist", "Image does not exist"

        image = dlib.load_rgb_image(image_path)
        faces = self.detect_face(image)

        if faces is None:
            return "NO FACE DETECTED", "NO FACE DETECTED", "NO FACE DETECTED"

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

            median_color_hair = calculate_median_color(upper_part)
            median_color_skin = calculate_median_color(img)

            # Vergleich der Medianfarben
            hair_diff = np.sqrt((median_color_hair[0] - median_color_skin[0]) ** 2 +
                                (median_color_hair[1] - median_color_skin[1]) ** 2 +
                                (median_color_hair[2] - median_color_skin[2]) ** 2)

            # Schwellwert für den Farbunterschied
            threshold = 80

            hair_hex = rgb_to_hex(median_color_hair)
            color_finder = ColorFinder()
            color_name = color_finder.find_color(median_color_skin)

            if hair_diff < threshold:
                return hair_hex, color_name, "Maybe bald or similar skin and hair color"
            else:
                return hair_hex, color_name, "Not bald or balding"


class EyeColorDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)
        self.flag = 0

    def detect_eye_color(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist", "Image does not exist"

        image = dlib.load_rgb_image(image_path)

        faces, gray, img_rgb = self.detect_face(image)

        if faces is None:
            return "NO FACE DETECTED", "NO FACE DETECTED"

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
            return "Eye region not found", "Eye region not found"

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

        median_array1 = tuple(array1)
        hex_color = rgb_to_hex(median_array1)

        color_finder = ColorFinder()
        color_name = color_finder.find_color(array1)
        return hex_color, color_name

    def detect_face(self, image):
        if image is None:
            return None, None, None

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
            return "NO FACE DETECTED", "NO FACE DETECTED", "NO FACE DETECTED"

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

# class BeardDetector(FaceDetector):
#     def __init__(self, predictor_path: str):
#         super().__init__(predictor_path)


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


# class GenderAgeDetector(FaceDetector):
#     def __init__(self, age_model, age_proto, gender_model, gender_proto, predictor_path: str):
#         super().__init__(predictor_path)
#         self.ageNet = cv2.dnn.readNet(age_model, age_proto)
#         self.genderNet = cv2.dnn.readNet(gender_model, gender_proto)
#         self.detector = dlib.get_frontal_face_detector()
#         self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#         self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
#         self.genderList = ['Male', 'Female']
#         self.padding = 20

#     def highlight_face(self, frame):
#         frame_opencv_dnn = frame.copy()
#         gray = cv2.cvtColor(frame_opencv_dnn, cv2.COLOR_BGR2GRAY)

#         faces = self.detector(gray)
#         face_boxes = []

#         for face in faces:
#             x1 = face.left()
#             y1 = face.top()
#             x2 = face.right()
#             y2 = face.bottom()
#             face_boxes.append([x1, y1, x2, y2])

#             shape = self.predictor(gray, face)

#             for i in range(0, 68):
#                 x = shape.part(i).x
#                 y = shape.part(i).y
#                 cv2.circle(frame_opencv_dnn, (x, y), 2, (0, 255, 0), -1)

#         return frame_opencv_dnn, face_boxes

#     def detect_age_gender(self, frame, face_box):
#         face = frame[max(0, face_box[1] - self.padding):
#                      min(face_box[3] + self.padding, frame.shape[0] - 1), max(0, face_box[0] - self.padding)
#                                                                           :min(face_box[2] + self.padding,
#                                                                                frame.shape[1] - 1)]

#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
#         self.genderNet.setInput(blob)
#         gender_preds = self.genderNet.forward()
#         gender = self.genderList[gender_preds[0].argmax()]

#         self.ageNet.setInput(blob)
#         age_preds = self.ageNet.forward()
#         age = self.ageList[age_preds[0].argmax()]

#         return gender, age

#     def process_image(self, image_path):
#         if not os.path.isfile(image_path):
#             return "Image does not exist", "Image does not exist"

#         frame = cv2.imread(image_path)
#         if frame is None:
#             print(f"Error: Unable to open image file {image_path}")
#             return

#         result_img, face_boxes = self.highlight_face(frame)
#         if not face_boxes:
#             return "No face detected", "No face detected"

#         for faceBox in face_boxes:
#             gender, age = self.detect_age_gender(frame, faceBox)
#             # print(f'Gender: {gender}')
#             # print(f'Age: {age[1:-1]} years')

#             # cv2.putText(result_img, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#             #             (0, 255, 255), 2, cv2.LINE_AA)

#             return gender, age
