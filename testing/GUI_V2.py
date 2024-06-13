import ttkbootstrap as ttk
import tkinter as tk
from PIL import Image, ImageTk
from video_landmarks import FaceLandmarkDetector

#Hallo anne (^-^)/

class GUI:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Capture")
        self.window.configure(
        bg="#f5f5f5")
        self.window.geometry("1000x700")

        predictor_path = "Utils/shape_predictor_68_face_landmarks.dat"
        self.face_landmark_detector = FaceLandmarkDetector(predictor_path)

        # Create 10x10 grid
        for i in range(10):
            self.window.grid_rowconfigure(i, weight=1)
            self.window.grid_columnconfigure(i, weight=1)

        # Place video_capture canvas in grid, spanning two rows and filling 3 columns on the right side
        self.canvas = tk.Canvas(self.window)
        self.canvas.grid(row=0, column=2, rowspan=6, columnspan=5, sticky="nsew", padx=20, pady=5)

        # Label Gender
        self.label_gender = ttk.Label(window, text="Gender", anchor='center', bootstyle="inverse-info")
        self.result_label_gender = ttk.Label(window, text="")
        self.label_gender.grid(row=0, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)
        self.result_label_gender.grid(row=1, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)

        self.label_eyes = ttk.Label(window, text="Augenfarbe", anchor='center', bootstyle="inverse-info")
        self.result_label_eyes = ttk.Label(window, text="")
        self.label_eyes.grid(row=2, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)
        self.result_label_eyes.grid(row=3, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)

        self.label_glasses = ttk.Label(window, text="Brille", anchor='center', bootstyle="inverse-info")
        self.result_label_glasses = ttk.Label(window, text="")
        self.label_glasses.grid(row=4, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)
        self.result_label_glasses.grid(row=5, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)

        self.label_facial = ttk.Label(window, text="Bart", anchor='center', bootstyle="inverse-info")
        self.result_label_facial = ttk.Label(window, text="")
        self.label_facial.grid(row=6, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)
        self.result_label_facial.grid(row=7, column=0, columnspan=2, sticky="nswe", padx=5, pady=5)

        # Create buttons
        self.display_button = ttk.Button(window, text="Display Result", bootstyle="success-outline",
                                         command=self.display_result)
        self.display_button.grid(row=8, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

        self.reset_button = ttk.Button(window, text="Reset", bootstyle="danger-outline", command=self.reset_result)
        self.reset_button.grid(row=9, column=0, columnspan=2, sticky="ew", padx=20, pady=10)

        self.kind_of_classifikation = 0
        self.update_frame()

    def update_frame(self):
        frame = self.face_landmark_detector.get_frame()
        if frame is not None:
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference to prevent garbage collection
        self.window.after(33, self.update_frame)  # Update every ~33 milliseconds (approx. 30 fps)

    @staticmethod
    def gender():
        # Implement the gender computation logic here
        return "Weiblich"

    @staticmethod
    def eyes():
        # Implement the eye color computation logic here
        return "Hat die Augenfarbe: XY"

    @staticmethod
    def glasses():
        # Implement the glasses computation logic here
        return "Trägt eine Brille"

    @staticmethod
    def facial():
        # Implement the facial hair computation logic here
        return "Trägt einen Bart"

    def display_result(self):
        self.result_label_gender.config(text=self.gender())
        self.result_label_eyes.config(text=self.eyes())
        self.result_label_glasses.config(text=self.glasses())
        self.result_label_facial.config(text=self.facial())

    def reset_result(self):
        self.result_label_gender.config(text="")
        self.result_label_eyes.config(text="")
        self.result_label_glasses.config(text="")
        self.result_label_facial.config(text="")


def main():
    root = tk.Tk()
    ttk.Style("flatly")
    GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
