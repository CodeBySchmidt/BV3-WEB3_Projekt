import dlib
import os
import cv2
from imutils import face_utils

from CountsPerSec import CountsPerSec
from VideoGet import VideoGet
from VideoShow import VideoShow

# Pfad zur shape_predictor_68_face_landmarks.dat im Utils-Ordner
dat_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))
predictor = dlib.shape_predictor(dat_file_path)

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()


def thread_both(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    # cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        # frame = putIterationsPerSec(frame, cps.countsPerSec())

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

        video_shower.frame = frame
        # cps.increment()


def main():
    thread_both()


if __name__ == "__main__":
    main()
