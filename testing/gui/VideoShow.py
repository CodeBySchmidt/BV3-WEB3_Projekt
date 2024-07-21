from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    # def show(self):
    #     while not self.stopped:
    #         # In a real application, you might want to add a delay here to control the frame rate
    #         pass  # Just keep the thread alive

    def stop(self):
        self.stopped = True
