import cv2
import imutils
import threading


class VideoCaptureWithFaceDetection:
    
    def __init__(self, video_source=0, kind_of_classifikation=0):
        dif_cascades = ['haarcascade_eye_tree_eyeglasses.xml', 'haarcascade_frontalface_default.xml']
        self.video_capture = cv2.VideoCapture(video_source)
        self.frame = None
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + dif_cascades[0])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + dif_cascades[1])
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()

    def _update_frame(self):
        while True:
            ret, frame = self.video_capture.read()
            if ret:
                frame = imutils.resize(frame, width=1000)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                eyes = self.eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in eyes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

    def get_frame(self):
        return self.frame

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
            
