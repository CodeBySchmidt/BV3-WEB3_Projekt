import cv2
import numpy as np
import os


class GlassesDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Haar-Cascade-Gesichtsdetektor
        self.net = cv2.dnn.readNetFromCaffe("Utils/face_detector/deploy.prototxt", "Utils/face_detector/opencv_face_detector.caffemodel")  # DNN auf dem ResNet-10 Modell
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Konvertieren in Graustufen
        self.lines = None

    # Zum erkennen, eines Gesichtes im Bild, mit Cascade Klassifizierer
    def detect_faces_with_cascade(self):
        faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    # Zum erkennen, einen Gesichtes im Bild, mit einem DNN (Deep Neural Network) Gesichtsdetektor von OpenCV, der auf dem Modell "ResNet-10" trainiert ist
    def detect_faces(self):
        blob = cv2.dnn.blobFromImage(cv2.resize(self.image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence-Schwellenwert einstellen
                box = detections[0, 0, i, 3:7] * np.array([self.image.shape[1], self.image.shape[0], self.image.shape[1], self.image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX - startX, endY - startY))
        return faces

    # Einzeichnen des erkannten Gesichtes, mit einem roten Rahmen
    @staticmethod
    def draw_faces(image, faces):
        # Durchlaufe die erkannten Gesichter
        for (x, y, w, h) in faces:
            # Zeichne ein Rechteck um das erkannte Gesicht
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)

    def detect_glasses(self, face):
        x, y, w, h = face
        img_blur = cv2.GaussianBlur(self.image, (11, 11), 50)
        roi_gray = img_blur[y:y + h, x:x + w]
        edges = cv2.Canny(roi_gray, 1, 150)

        cv2.imshow("Glasses", edges)

        # Überprüfen, ob horizontale Kanten im Bild vorhanden sind
        vertical_edges = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=9)
        vertical_edges = cv2.convertScaleAbs(vertical_edges)
        _, vertical_edges = cv2.threshold(vertical_edges, 1, 255, cv2.THRESH_BINARY)

        # Suche nach vertikalen Kanten
        contours, _ = cv2.findContours(vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Zeichne die Konturen um das Gesicht
        for contour in contours:
            x_contour, y_contour, w_contour, h_contour = cv2.boundingRect(contour)
            cv2.rectangle(self.image, (x + x_contour, y + y_contour),
                          (x + x_contour + w_contour, y + y_contour + h_contour), (0, 255, 0), 2)

        if len(contours) > 0:
            return True  # Brille erkannt
        else:
            return False  # Keine Brille erkannt

    # def draw_glasses(self, face):
    #     return

    def display_result(self):
        faces = self.detect_faces()
        # self.draw_faces(self.image, faces)
        for face in faces:
            if self.detect_glasses(face):  # If Abfrage ob detect_glasses "True" zurück gibt -> "if self.detect_glasses(face) == True"
                # self.draw_glasses(face)  # Wenn es true ist, wird die draw_glasses aufgerufen und zeichnet die Brille ein
                print("Brille erkannt!")
            else:
                print("Brille nicht erkannt!")
        self.resize_image(500)

    def resize_image(self, desired_width):
        height, width, _ = self.image.shape
        ratio = width / height
        desired_height = int(desired_width / ratio)
        resized_image = cv2.resize(self.image, (desired_width, desired_height))
        cv2.imshow('Resized Image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = 'img/glasses/3.jpg'
    glasses_detector = GlassesDetector(image_path)
    glasses_detector.display_result()

    # # Eigenschaften die im Objekt "GlassesDetector vorhanden sind
    # path = glasses_detector.image_path
    # img = glasses_detector.image
    # gray_img = glasses_detector.gray
    # lines = glasses_detector.lines
    #
    # # Mögliche Ausgabe zum Überprüfen der Eigenschaften
    # print(path)
    # cv2.imshow("Image", img)
    # cv2.imshow("Gray Image", gray_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(lines)
