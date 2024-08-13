import os
import threading
import time
import numpy as np
import dlib
import cv2
import webcolors
from imutils import face_utils
import matplotlib.pyplot as plt


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

    def detect_glasses(self, img):
        faces = self.detect_face(img)
        if faces is None:
            return "No face detected"

        rect = faces[0]
        img_cropped = self.detect_nose_region(img, rect)
        img_blur = cv2.GaussianBlur(img_cropped, (9, 9), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        edges_center = edges.T[(int(len(edges.T) / 2))]

        if 255 in edges_center:
            return True
        else:
            return False

    def display_results(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist"

        img = dlib.load_rgb_image(image_path)
        glasses = self.detect_glasses(img)

        if glasses == "No face detected":
            return "No face detected"
        elif glasses == True:
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
    if len(image.shape) == 3:
        median_b = int(np.median(image[:, :, 0].flatten()))
        median_g = int(np.median(image[:, :, 1].flatten()))
        median_r = int(np.median(image[:, :, 2].flatten()))
        hex_color = '#{:02x}{:02x}{:02x}'.format(median_r, median_g, median_b)
        return (median_b, median_g, median_r), hex_color
    else:
        median_value = int(np.median(image.flatten()))
        hex_color = '#{:02x}{:02x}{:02x}'.format(median_value, median_value, median_value)
        return (median_value, median_value, median_value), hex_color


class HairColorDetector(FaceDetector):
    def __init__(self, predictor_path: str):
        super().__init__(predictor_path)

    def crop_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        if rects:
            rect = rects[0]
            landmarks = self.predictor(gray, rect)
            face_utils.shape_to_np(landmarks)
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            # Berechnung der Bounding Box
            x1 = max(0, x)
            y1 = max(0, y - h // 2)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

            cropped_image = image[y1:y2, x1:x2]
            img = image[y:y + h, x:x + w]

            # Oberer Teil des Gesichts
            upper_part = cropped_image[:int(cropped_image.shape[0] * 0.25), :]

            return upper_part, img
        else:
            return None, None

    def find_hair_color(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist"

        image = dlib.load_rgb_image(image_path)
        upper_part, img = self.crop_face(image)

        median_color_hair, hair_hex = calculate_median_color(upper_part)
        median_color_skin, skin_hex = calculate_median_color(img)

        color_finder = ColorFinder()
        color_name = color_finder.find_color(median_color_hair)

        # Vergleich der Medianfarben
        hair_diff = np.sqrt((median_color_hair[0] - median_color_skin[0]) ** 2 +
                            (median_color_hair[1] - median_color_skin[1]) ** 2 +
                            (median_color_hair[2] - median_color_skin[2]) ** 2)
        # print(f"Unterschied der Farbwerte von Haar und Haut: {hair_diff}")

        # Schwellwert für den Farbunterschied
        threshold = 60

        if hair_diff < threshold:
            return color_name, "Probably a bald spot or a bald head"
        else:
            return color_name, "Possible not bald or balding"


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

        array1 = roi_eye[(row // 2):(row // 2) + 1, int((col // 3) + 2):int((col // 3)) + 9]
        array1 = array1[0][5]
        array1 = tuple(array1)

        color_name = self.find_color(array1)
        # print(color_name)

        # detected_color = np.zeros((100, 300, 3), dtype=np.uint8)
        # detected_color[:] = array1[::1]
        #
        # # cv2.putText(detected_color, color_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        # # cv2.LINE_AA)
        #
        # roi_top = (row // 2)
        # roi_bottom = (row // 2) + 1
        # roi_left = (col // 3) + 2
        # roi_right = (col // 3) + 9
        # new_rgb = (0, 0, 255)
        # roi_eye[roi_top:roi_bottom, roi_left:roi_right] = new_rgb
        #
        # # cv2.imshow("Detected Color", detected_color)
        # # cv2.imshow("frame", roi_eye)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # roi_eye_rgb = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2RGB)
        # detected_color_rgb = cv2.cvtColor(detected_color, cv2.COLOR_BGR2RGB)
        # plt.imshow(detected_color_rgb)
        # plt.title("Detected Color")
        # plt.show()
        # plt.imshow(roi_eye_rgb)
        # plt.title("Eye Detected")
        # plt.show()
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


class GenderAgeDetector(FaceDetector):
    def __init__(self, age_model, age_proto, gender_model, gender_proto, predictor_path: str):
        super().__init__(predictor_path)
        self.ageNet = cv2.dnn.readNet(age_model, age_proto)
        self.genderNet = cv2.dnn.readNet(gender_model, gender_proto)
        self.detector = dlib.get_frontal_face_detector()
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']
        self.padding = 20

    def highlight_face(self, frame):
        frame_opencv_dnn = frame.copy()
        gray = cv2.cvtColor(frame_opencv_dnn, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        face_boxes = []

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            face_boxes.append([x1, y1, x2, y2])

            shape = self.predictor(gray, face)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                cv2.circle(frame_opencv_dnn, (x, y), 2, (0, 255, 0), -1)

        return frame_opencv_dnn, face_boxes

    def detect_age_gender(self, frame, face_box):
        face = frame[max(0, face_box[1] - self.padding):
                     min(face_box[3] + self.padding, frame.shape[0] - 1), max(0, face_box[0] - self.padding)
                                                                          :min(face_box[2] + self.padding,
                                                                               frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.genderNet.setInput(blob)
        gender_preds = self.genderNet.forward()
        gender = self.genderList[gender_preds[0].argmax()]

        self.ageNet.setInput(blob)
        age_preds = self.ageNet.forward()
        age = self.ageList[age_preds[0].argmax()]

        return gender, age

    def process_image(self, image_path):
        if not os.path.isfile(image_path):
            return "Image does not exist", "Image does not exist"

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to open image file {image_path}")
            return

        result_img, face_boxes = self.highlight_face(frame)
        if not face_boxes:
            return "No face detected", "No face detected"

        for faceBox in face_boxes:
            gender, age = self.detect_age_gender(frame, faceBox)
            # print(f'Gender: {gender}')
            # print(f'Age: {age[1:-1]} years')

            # cv2.putText(result_img, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #             (0, 255, 255), 2, cv2.LINE_AA)

            return gender, age


class BeardDetector:

    def init(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

# Alter Code
#
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
