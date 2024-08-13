# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import os

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = dat_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:


            # draws lines between the facial landmarks
            cv2.line(image, (shape[1][0], shape[1][1]), (shape[31][0], shape[31][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[2][0], shape[2][1]), (shape[1][0], shape[1][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[31][0], shape[31][1]), (shape[28][0], shape[28][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[28][0], shape[28][1]), (shape[35][0], shape[35][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[35][0], shape[35][1]), (shape[15][0], shape[15][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[15][0], shape[15][1]), (shape[14][0], shape[14][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[14][0], shape[14][1]), (shape[13][0], shape[13][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[13][0], shape[13][1]), (shape[12][0], shape[12][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[12][0], shape[12][1]), (shape[11][0], shape[11][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[11][0], shape[11][1]), (shape[10][0], shape[10][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[10][0], shape[10][1]), (shape[9][0], shape[9][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[9][0], shape[9][1]), (shape[8][0], shape[8][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[8][0], shape[8][1]), (shape[7][0], shape[7][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[7][0], shape[7][1]), (shape[6][0], shape[6][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[6][0], shape[6][1]), (shape[5][0], shape[5][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[5][0], shape[5][1]), (shape[4][0], shape[4][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[4][0], shape[4][1]), (shape[3][0], shape[3][1]), (0, 255, 0), 2)
            cv2.line(image, (shape[3][0], shape[3][1]), (shape[2][0], shape[2][1]), (0, 255, 0), 2)

            # cv2.polylines(image,
            #              (shape[2][0], shape[2][1]),
            #              (shape[8][0], shape[8][1]),
            #              (shape[14][0], shape[14][1]),
            #              (shape[29][0], shape[29][1]),
            #              (0, 255, 0), 2)

        # draws circle with labels
        for idx, (x, y) in enumerate(shape):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # show the output image with the face detections + facial landmarks
    # esc for exit!
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF

    if k == 32:
        cv2.imwrite("test_frame.jpg", image)

    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()
