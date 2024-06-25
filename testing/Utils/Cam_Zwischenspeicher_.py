from vmbpy import *
import cv2

class FrameHandler:
    def __init__(self):
        self.running = True

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        # Konvertiere das Frame in ein OpenCV-Bild
        image = frame.as_opencv_image()

        # Zeige das Bild in einem OpenCV-Fenster an
        cv2.imshow('Live Stream', image)

        # Gib das Frame wieder frei
        cam.queue_frame(frame)

def main():
    with VmbSystem.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            print("Keine Kameras gefunden!")
            return

        with cams[0] as cam:
            frame_handler = FrameHandler()

            # Starte das Streaming
            cam.start_streaming(frame_handler)

            while frame_handler.running:
                # Warte auf eine Tasteneingabe und beende, wenn 'q' gedrückt wird
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    frame_handler.running = False
                    cam.stop_streaming()

            # Stoppe das Streaming
            cam.stop_streaming()

            # Schließe das OpenCV-Fenster
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()