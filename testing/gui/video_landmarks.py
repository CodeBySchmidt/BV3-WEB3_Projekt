import threading
import time
import numpy as np
import dlib
import cv2
import webcolors_print
import matplotlib.pyplot as plt
from imutils import face_utils


class FaceDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)


class GlassesDetector(FaceDetector):
    def __init__(self, predictor_path):
        super().__init__(predictor_path)

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

        return img_cropped

    def detect_glasses(self, img):
        img_cropped = self.detect_nose_region(img)
        img_blur = cv2.GaussianBlur(img_cropped, (9, 9), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        plt.imshow(img_blur)
        plt.title("Blur Image")
        plt.show()

        plt.imshow(edges, cmap='gray')
        plt.title("Edges Image")
        plt.show()

        edges_center = edges.T[(int(len(edges.T) / 2))]

        if 255 in edges_center:
            return True
        else:
            return False

    def display_results(self, img):
        glasses = self.detect_glasses(img)
        if glasses:
            return "Trägt Brille"
        else:
            return "Keine Brille"


class EyeColorDetector:
    def __init__(self, predictor_path):
        self.flag = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

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

    def detect_eye_color(self, img_path):
        img_rgb = cv2.imread(img_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        dlib_faces = self.detector(gray, 0)

        for face in dlib_faces:
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

        x = roi_eye.shape
        row = x[0]
        col = x[1]

        array1 = roi_eye[(row // 2):(row // 2) + 1, int((col // 3) + 2):int((col // 3)) + 9]
        array1 = array1[0][5]
        array1 = tuple(array1)

        color_name = self.find_color(array1)
        # print(color_name)

        detected_color = np.zeros((100, 300, 3), dtype=np.uint8)
        detected_color[:] = array1[::1]

        # cv2.putText(detected_color, color_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        # cv2.LINE_AA)

        roi_top = (row // 2)
        roi_bottom = (row // 2) + 1
        roi_left = (col // 3) + 2
        roi_right = (col // 3) + 9
        new_rgb = (0, 0, 255)
        roi_eye[roi_top:roi_bottom, roi_left:roi_right] = new_rgb

        # cv2.imshow("Detected Color", detected_color)
        # cv2.imshow("frame", roi_eye)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        roi_eye_rgb = cv2.cvtColor(roi_eye, cv2.COLOR_BGR2RGB)
        detected_color_rgb = cv2.cvtColor(detected_color, cv2.COLOR_BGR2RGB)
        plt.imshow(detected_color_rgb)
        plt.title("Detected Color")
        plt.show()
        plt.imshow(roi_eye_rgb)
        plt.title("Eye Detected")
        plt.show()
        return color_name


class FaceLandmarkDetector:
    def __init__(self, predictor_path, camera_index=0):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Kamera konnte nicht geöffnet werden. Überprüfen Sie den Kamera-Index und ob die Kamera angeschlossen "
                "ist.")

    def detect_and_draw_landmarks(self, image, draw_circles=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Nur Kreise zeichnen, wenn draw_circles True ist
            if draw_circles:
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        return image

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_with_circles = self.detect_and_draw_landmarks(frame)
        return cv2.cvtColor(frame_with_circles, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter

    def save_screenshot(self, file_path):
        ret, frame = self.cap.read()
        if ret:
            frame_without_circles = self.detect_and_draw_landmarks(frame, draw_circles=False)
            cv2.imwrite(file_path, frame_without_circles)
            print(f"Screenshot saved to {file_path}")
        else:
            print("Failed to capture image.")

    def take_screenshot(self, file_path):
        time.sleep(5)  # Wait for 5 seconds
        self.save_screenshot(file_path)

    def show_live_video(self):
        screenshot_path = "screenshot.jpg"
        screenshot_thread = threading.Thread(target=self.take_screenshot, args=(screenshot_path,))
        screenshot_thread.start()

        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret or time.time() - start_time > 6:
                break
            frame_with_circles = self.detect_and_draw_landmarks(frame)
            cv2.imshow("Live Video - Press ESC to exit", frame_with_circles)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:  # ESC key to break
                break

        screenshot_thread.join()  # Warten Sie darauf, dass der Screenshot-Thread abgeschlossen ist
        self.cap.release()
        cv2.destroyAllWindows()

    def update_image(self, img_element):
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            # Convert the frame to a format suitable for display in niceGUI
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            img_element.source = f'data:image/jpeg;base64,{buffer.tobytes().encode("base64").decode()}'
            time.sleep(0.1)


# if __name__ == "__main__":
#     try:
#         predictor_path = "Utils/shape_predictor_68_face_landmarks.dat"
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
