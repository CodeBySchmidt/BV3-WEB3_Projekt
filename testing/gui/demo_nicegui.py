import base64
import signal
import time

import cv2
import numpy as np
from fastapi import Response

from nicegui import Client, app, core, ui
import dlib
from imutils import face_utils

import webcolors

from skript import *

import os

#
# ________________________________________________________________________________________________________
#
#  Einlesen des Shape Predictors und initialize face detector
#
# ________________________________________________________________________________________________________
#

def load_shape_predictor(dat_file_path):
    # Überprüfen, ob der Pfad korrekt ist und die Datei existiert
    if not os.path.exists(dat_file_path):
        print(f"Datei nicht gefunden: {dat_file_path}")
        return None

    try:
        # Laden des Prädiktors mit dlib
        predictor = dlib.shape_predictor(dat_file_path)
        print("Prädiktor erfolgreich geladen")
        return predictor
    except RuntimeError as e:
        print(f"Fehler beim Laden des Prädiktors: {e}")
        return None

# Pfad zur shape_predictor_68_face_landmarks.dat im Utils-Ordner
dat_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))

# Überprüfen des berechneten Pfads
print(f"Berechneter Pfad: {dat_file_path}")

# Aufrufen der Funktion mit dem Dateipfad, wenn alles klappt, dann wurde der shape_predictor geladen
predictor = load_shape_predictor(dat_file_path)

# Jetzt kann der `predictor` weiterverwenden, falls er erfolgreich geladen wurde
if predictor:
    print("Bereit zur Verwendung des Prädiktors.")
else:
    print("Prädiktor konnte nicht geladen werden.")

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()


# ________________________________________________________________________________________________________
#
#  Globale Variblen und Methoden zum einzeichnen und zum einbinden in der GUI
#
# ________________________________________________________________________________________________________
#

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')
# OpenCV is used to access the webcam.
video_capture = cv2.VideoCapture(0)

frame_counter = 0


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def detect_and_draw_landmarks(frame, draw_landmarks=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if draw_landmarks:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


@app.get('/video/frame')
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_video_frame() -> Response:
    global frame_counter
    ret, frame = video_capture.read()
    if not ret:
        return placeholder
    frame_counter += 1

    if frame_counter % 50 == 0:
        cv2.imwrite("current_frame.jpg", detect_and_draw_landmarks(frame, False))
        jpeg = convert(detect_and_draw_landmarks(frame))
        print(frame_counter)
        frame_counter = 0
        return Response(content=jpeg, media_type='image/jpeg')
    else:
        jpeg = convert(detect_and_draw_landmarks(frame))
        return Response(content=jpeg, media_type='image/jpeg')


# Callback-Funktion für Button-Klick
def button_clicked():

    facial_hair_result = facial_hair()
    facial_hair_color_result = facial_hair_color()
    glasses_result = glasses()
    hair_color_result = hair_color()
    eye_color_result = eye_color()
    race_result = race()
    gender_age_result = gender_age()


    # Label aktualisieren, um den neuen Wert anzuzeigen
    facial_hair_label.set_text(facial_hair_result)
    facial_hair_color_label.set_text(facial_hair_color_result)
    glasses_label.set_text(glasses_result)
    hair_color_label.set_text(hair_color_result)
    eye_color_label.set_text(eye_color_result)
    race_label.set_text(race_result)
    gender_label.set_text(gender_age_result[0])
    age_label.set_text(gender_age_result[1])

    output_1_color.style(f"background-color: {get_color_name_or_hex(eye_color_result)} !important;")
    output_1_color.set_text(eye_color_result)

    output_2_color.style(f"background-color: {get_color_name_or_hex(facial_hair_color_result)} !important;")
    output_2_color.set_text(facial_hair_color_result)

    output_3_color.style(f"background-color: {get_color_name_or_hex(hair_color_result)} !important;")
    output_3_color.set_text(hair_color_result)


def get_color_name_or_hex(input_value):
    # Wenn der input_value ein Hex ist, dann wird dieser zu einem Name umgewandelt
    if input_value.startswith('#'):
        # Hex wird klein geschieben, keine großbuchstaben soll da sein, damit man diesen vergleichen kann
        hex_color = input_value.lower()

        if hex_color in webcolors.CSS3_HEX_TO_NAMES:
            color_name = webcolors.CSS3_HEX_TO_NAMES[hex_color]
            # print(f"Hex: {hex_color} passender Name: {color_name}")
            return color_name
        else:
            print(f"Hex: {hex_color} ist nicht findbar in CSS3_HEX_TO_NAMES.")
    else:
        # Wenn der input_value ein Name ist, dann wird dieser zu einem Hex umgewandelt
        color_name = input_value.lower()
        if color_name in webcolors.CSS3_NAMES_TO_HEX:
            hex_color = webcolors.CSS3_NAMES_TO_HEX[color_name]
            # print(f"Name: {color_name} passender Hex: {hex_color}")
            return hex_color
        else:
            print(f"Name {color_name} ist nicht findbar in CSS3_NAMES_TO_HEX.")


#
# ________________________________________________________________________________________________________
#
#  benutzerdefinierte CSS-Klassen
#
# ________________________________________________________________________________________________________
#

# Definiere benutzerdefinierte CSS-Klassen für die Farben
ui.add_css('''

    .primary { background-color: #262626; }
    .secondary { background-color: #424242; }
    
    .btn {
        font-weight: bold;
        border-radius: 15px; /* Rounded corners (assuming large value for roundness) */
        color: white;
        font-size: 24px;
        padding: 20px 36px;
        transition: 0.3s ease-out;
    }

    .btn-blue {
        background-color: #3491ED !important;
    }

    .btn-blue:hover {
        background-color: #0ea5e9 !important;
    }
    
    .inset-shadow {
        box-shadow: 0px 0px 10px 5px rgba(0, 0, 0, 0.50) inset;
    }
    
    .shadow-color {
        box-shadow: 8px 8px 10px 2px rgba(0, 0, 0, 0.50);
    }
    
''')

#
# ________________________________________________________________________________________________________
#
#  GUI Aufbau
#
# ________________________________________________________________________________________________________
#


ui.query("body").classes("primary")

# Erstelle das Grid mit den farbigen Labels
with ui.grid(rows=6, columns=16).classes("gap-5 w-full p-6").style("height: 95vh;"):

    with ui.element("container_logo").classes("row-start-1 row-span-1 col-span-5 h-auto rounded secondary flex justify-center items-center overflow-hidden"):
        ui.label("Logo").classes("text-white flex justify-center items-center text-4xl")

    with ui.element("container_outputs").classes("row-start-2 row-span-4 col-span-5 rounded"):
        with ui.grid(rows=4, columns=2).classes("h-full w-full gap-5"):
            # Erste Zeile
            with ui.element("container_output_1").classes("rounded col-span-2 row-start-1 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_1 links
                with ui.element("output_1").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    eye_color_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Eye Color").classes("text-lg").style("color: white;")

                # output_1_color rechts
                with ui.element("output_1_color").classes("col-start-2 col-span-1 row-start-1 row-span-1 flex justify-center items-center"):
                    output_1_color = ui.label("None").classes("text-black bg-white p-5 rounded flex justify-center items-center inset-shadow").style("height: 80%; width: 80%;")

            # Zweite Zeile
            with ui.element("container_output_2").classes("rounded col-span-2 row-start-2 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_2 links
                with ui.element("output_2").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    facial_hair_color_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Facial Hair Color").classes("text-lg").style("color: white;")

                # output_2_color rechts
                with ui.element("output_2_color").classes("col-start-2 col-span-1 row-start-1 row-span-1 flex justify-center items-center"):
                    output_2_color = ui.label("None").classes("text-black bg-white p-5 rounded flex justify-center items-center inset-shadow").style("height: 80%; width: 80%")

            # Dritte Zeile
            with ui.element("container_output_3").classes("rounded col-span-2 row-start-3 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_3 links
                with ui.element("output_3").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    hair_color_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Hair Color").classes("text-lg").style("color: white;")
                # output_3_color rechts
                with ui.element("output_3_color").classes("col-start-2 col-span-1 row-start-1 row-span-1 flex justify-center items-center"):
                    output_3_color = ui.label("None").classes("bg-white text-black p-5 rounded flex justify-center items-center inset-shadow").style("height: 80%; width: 80%")

            # Vierte Zeile # output_4 links
            with ui.element("output_4").classes("col-start-1 col-span-1 row-start-4 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                facial_hair_label = ui.label("None").classes("text-4xl").style("color: white;")
                ui.label("Facial Hair").classes("text-lg").style("color: white;")

            # Vierte Zeile # output_5 rechts
            with ui.element("output_5").classes("col-start-2 col-span-1 row-start-4 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                glasses_label = ui.label("None").classes("text-4xl").style("color: white;")
                ui.label("Glasses").classes("text-lg").style("color: white;")

    with ui.element("container_camera").classes("col-span-11 row-start-1 row-span-5 h-auto rounded p-1 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
        video_image = ui.interactive_image().classes("h-full w-full object-contain")

    with ui.element("container_ki").classes("col-span-12 row-start-6 rounded pr-6 overflow-hidden"):
        with ui.grid(rows=1, columns=3).classes("h-fit w-full gap-5"):
            # Erste Spalte -> Age
            with ui.element("container_output_age").classes("rounded col-span-4 row-start-1 grid grid-cols-2 gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_age
                with ui.element("output_age").classes(
                        "col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    age_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Age").classes("text-lg").style("color: white;")

            # Zweite Spalte -> Race
            with ui.element("container_output_race").classes("rounded col-span-4 row-start-1 grid grid-cols-2 gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_race
                with ui.element("output_race").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    race_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Race").classes("text-lg").style("color: white;")

            # Dritte Spalte -> Gender
            with ui.element("container_output_gender").classes("rounded col-span-4 row-start-1 grid grid-cols-2 gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_gender
                with ui.element("output_gender").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    gender_label = ui.label("None").classes("text-4xl").style("color: white;")
                    ui.label("Gender").classes("text-lg").style("color: white;")

    with ui.element("container_button").classes("col-span-4 row-start-6 flex justify-end items-center overflow-hidden"):
        ui.button("Button", on_click=button_clicked).classes("btn btn-blue")

# A timer constantly updates the source of the image.
# Because data from same paths are cached by the browser,
# we must force an update by adding the current timestamp to the source.
ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))


async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


def handle_sigint(signum, frame) -> None:
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    video_capture.release()


app.on_shutdown(cleanup)

# We also need to disconnect clients when the app is stopped with Ctrl+C,
# because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
signal.signal(signal.SIGINT, handle_sigint)

ui.run(title="Test")
