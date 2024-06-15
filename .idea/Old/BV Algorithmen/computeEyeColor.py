import cv2 as cv2


class computeEyeColor:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.eye_colors = {
            "blue": 0,
            "green": 0,
            "brown": 0,
            "gray": 0
        }
        self.eye_color = None
        self.face_color = None
        self.eye_color = None
        
        
        class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
        EyeColor = {
            class_name[0] : ((166, 21, 50), (240, 100, 85)),
            class_name[1] : ((166, 2, 25), (300, 20, 75)),
            class_name[2] : ((2, 20, 20), (40, 100, 60)),
            class_name[3] : ((20, 3, 30), (65, 60, 60)),
            class_name[4] : ((0, 10, 5), (40, 40, 25)),
            class_name[5] : ((60, 21, 50), (165, 100, 85)),
            class_name[6] : ((60, 2, 25), (165, 20, 65))
        }
        
#