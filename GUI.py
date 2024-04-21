import tkinter as tk
import ttkbootstrap as ttk
from PIL import Image, ImageTk
from camera_capture import VideoCaptureWithFaceDetection


class GUI:

    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Video Capture")
        self.window.configure(bg="#f5f5f5")  # Set window background color to black
        self.window.geometry("1200x800")  # Set window size
        self.kind_of_classifikation = 0
        self.video_capture = VideoCaptureWithFaceDetection(video_source, kind_of_classifikation=self.kind_of_classifikation)

        # Create 5x5 grid
        for i in range(5):
            self.window.grid_rowconfigure(i, weight=1)
            self.window.grid_columnconfigure(i, weight=1)

        # Place video_capture canvas in grid, spanning two rows and filling 3 columns on the right side
        self.canvas = tk.Canvas(self.window)
        self.canvas.grid(row=0, column=1, rowspan=5, columnspan=5, sticky="nsew", padx=20, pady=20)

        # Place labels on the left side in the first column
        labels_text = ["Alter:", "Haarfarbe: ", "Augenfarbe:", "Geschlecht (?): ", "Bart:"]
        for i, label_text in enumerate(labels_text):
            frame = ttk.Frame(master=window)
            label = ttk.Label(window, text=label_text, bootstyle="inverse-info")
            frame.grid(row=i, column=0, sticky="nsew", padx=20, pady=20)
            label.grid(row=i, column=0, sticky="nwe", padx=20, pady=20)

        self.button = ttk.Button(window, text="Switch", bootstyle="danger-outline", command=self.switch)
        self.button.grid(row=5, column=0, columnspan=2, sticky="ew", padx=20, pady=30)
    
        # Place button in the last column on the right side
        self.button = ttk.Button(window, text="Compute", bootstyle="danger-outline", command=self.compute)
        self.button.grid(row=5, column=3, columnspan=2, sticky="ew", padx=20, pady=30)

        self.update_frame()

    def update_frame(self):
        frame = self.video_capture.get_frame()
        if frame is not None:
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # keep a reference to prevent garbage collection
        self.window.after(33, self.update_frame)  # Update every ~33 milliseconds (approx. 30 fps)
    
    def switch(self, video_source=0):
        if self.kind_of_classifikation == 0:
            self.kind_of_classifikation = 1
        else:
            self.kind_of_classifikation = 0
        self.video_capture = VideoCaptureWithFaceDetection(video_source, kind_of_classifikation=self.kind_of_classifikation)
        
    
    def compute(self):
        "Alter:" 
        "Haarfarbe: " 
        "Augenfarbe:"
        self.computeEyeColor()
        "Geschlecht (?): "
        "Bart:"
        


def main():
    root = tk.Tk()
    ttk.Style("flatly")
    GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
