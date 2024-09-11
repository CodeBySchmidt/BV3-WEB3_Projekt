import asyncio
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


# Pfad zur shape_predictor_68_face_landmarks.dat im Utils-Ordner
dat_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'Utils', 'shape_predictor_68_face_landmarks.dat'))
predictor = dlib.shape_predictor(dat_file_path)

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()

# ________________________________________________________________________________________________________
#
#  Globale Variablen und Methoden zum einzeichnen und zum einbinden in der GUI
#
# ________________________________________________________________________________________________________
#

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')
# OpenCV is used to access the webcam.
video_capture = cv2.VideoCapture(0)

draw_landmarks_state = False
state = True


@app.get('/video/frame')
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_video_frame() -> Response:
    global frame_counter
    ret, frame = video_capture.read()
    if not ret:
        return placeholder

    if draw_landmarks_state:
        jpeg = convert(detect_and_draw_landmarks(frame, draw_landmarks=draw_landmarks_state))
        return Response(content=jpeg, media_type="image/jpeg")
    else:
        jpeg = convert(frame)
        return Response(content=jpeg, media_type="image/jpeg")


async def image_processing():
    await asyncio.sleep(0.1)  # Async sleep anstelle von blockierendem sleep

    # Haarfarbe und Typ
    print("Creating task for hair_color function")
    try:
        hair_color_task = asyncio.create_task(hair_color())
        hair_color_and_typ = await hair_color_task
        print(f"hair_color_and_typ result: {hair_color_and_typ}")
        print("_________________________________________________________________________________________________")
        print()
    except Exception as e:
        print(f"Error in hair_color function: {e}")

    # Augenfarbe
    print("Creating task for eye_color function")
    try:
        eye_color_task = asyncio.create_task(eye_color())
        eye_color_result = await eye_color_task
        print(f"eye_color result: {eye_color_result}")
        print("_________________________________________________________________________________________________")
        print()
    except Exception as e:
        print(f"Error in eye_color function: {e}")

    # Brille
    print("Creating task for glasses function")
    try:
        glasses_task = asyncio.create_task(glasses())
        glasses_result = await glasses_task
        print(f"glasses result: {glasses_result}")
        print("_________________________________________________________________________________________________")
        print()
    except Exception as e:
        print(f"Error in glasses function: {e}")

    hair_typ_label.set_text(hair_color_and_typ[1])
    hair_color_label.set_text(hair_color_and_typ[0])
    output_3_color.style(f"background-color: {hair_color_and_typ[0]} !important;")

    eye_color_label.set_text(eye_color_result)
    glasses_label.set_text(glasses_result)


async def neural_networks():
    await asyncio.sleep(0.1)

    # FACIAL HAIR TASK
    # try:
    #     print("Starting facial_hair function")
    #     facial_hair_task = asyncio.create_task(facial_hair())
    #     facial_hair_result = await facial_hair_task
    #     print(f"facial_hair result: {facial_hair_result}")
    #     print("_________________________________________________________________________________________________")
    #     print()
    # except Exception as e:
    #     print(f"Error in neural_networks - facial_hair: {e}")

    # AGE GENDER RACE TASK
    try:
        print("Starting age_gender_race function")
        age_gender_race_task = asyncio.create_task(gender_age_race())
        age_gender_race_result = await age_gender_race_task
        print(f"age_gender race result: {age_gender_race_result}")
        print("_________________________________________________________________________________________________")
        print()
    except Exception as e:
        print(f"Error in neural_networks - age_gender_race: {e}")

    # facial_hair_label.set_text(facial_hair_result)
    age_label.set_text(age_gender_race_result[0])
    gender_label.set_text(age_gender_race_result[1])
    race_label.set_text(age_gender_race_result[2])


async def button_clicked():
    global state

    ret, frame = video_capture.read()
    if ret:
        screenshot_path = "current_frame.jpg"
        cv2.imwrite(screenshot_path, frame)
    else:
        print("Failed to capture frame")

    if state:
        video_image.set_visibility(False)

    await asyncio.create_task(image_processing())
    await asyncio.create_task(neural_networks())

    video_image.set_visibility(True)


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def detect_and_draw_landmarks(frame, draw_landmarks: bool):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        if draw_landmarks:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


def on_checkbox_change(value):
    global draw_landmarks_state

    if value:
        draw_landmarks_state = not draw_landmarks_state


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
            return
            # print(f"Name {color_name} ist nicht findbar in CSS3_NAMES_TO_HEX.")


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
with ui.grid(rows=6, columns=16).classes("gap-5 w-full p-5").style("height: 95vh;"):
    # Erste Zeile -> Logo
    with ui.element("container_logo").classes(
            "row-start-1 row-span-1 col-span-5 h-auto rounded secondary flex justify-center items-center overflow-hidden"):
        ui.label("Logo").classes("text-white flex justify-center items-center text-3xl")

    with ui.element("container_outputs").classes("row-start-2 row-span-4 col-span-5 rounded"):
        with ui.grid(rows=4, columns=2).classes("h-full w-full gap-5"):
            # Erste Zeile Hair Color
            with ui.element("container_output_1").classes(
                    "rounded col-span-2 row-start-1 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_3 links
                with ui.element("output_3").classes(
                        "col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    hair_color_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Hair Color").classes("text-lg").style("color: white;")
                # output_3_color rechts
                with ui.element("output_3_color").classes(
                        "col-start-2 col-span-1 row-start-1 row-span-1 flex justify-center items-center"):
                    output_3_color = ui.label().classes(
                        "bg-white text-black p-5 rounded flex justify-center items-center inset-shadow").style(
                        "height: 80%; width: 80%")

            # Zweite Zeile -> Hair Type
            with ui.element("container_output_2").classes(
                    "rounded col-span-2 row-start-2 grid gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_2 links
                with ui.element("output_2").classes(
                        "col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    hair_typ_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Hair Type").classes("text-lg").style("color: white;")

            # Dritte Zeile -> Eye Color
            with ui.element("container_output_3").classes(
                    "rounded col-span-2 row-start-3 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_1 links
                with ui.element("output_1").classes(
                        "col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    eye_color_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Eye Color").classes("text-lg").style("color: white;")

                # output_1_color rechts
                with ui.element("output_1_color").classes(
                        "col-start-2 col-span-1 row-start-1 row-span-1 flex justify-center items-center"):
                    output_1_color = ui.label().classes(
                        "text-black bg-white p-5 rounded flex justify-center items-center inset-shadow").style(
                        "height: 80%; width: 80%;")

            # Vierte Zeile # output_4 links -> Facial Hair
            with ui.element("output_4").classes(
                    "col-start-1 col-span-1 row-start-4 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                facial_hair_label = ui.label("None").classes("text-3xl").style("color: white;")
                ui.label("Facial Hair").classes("text-lg").style("color: white;")

            # Zweite Zeile # output_5 rechts -> Glasses
            with ui.element("output_5").classes(
                    "col-start-2 col-span-1 row-start-4 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                glasses_label = ui.label("None").classes("text-3xl").style("color: white;")
                ui.label("Glasses").classes("text-lg").style("color: white;")

    with ui.element("container_camera").classes(
            "container_camera col-span-11 row-start-1 row-span-5 h-auto flex justify-center items-center rounded p-1 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
        video_image = ui.interactive_image().classes("h-full w-full object-contain")
        camera_label = ui.label("Calculating...").classes("text-3xl").style("color: white;")
        ui.spinner(size="4em", thickness="5").classes("m-5")

    with ui.element("container_ki").classes("col-span-12 row-start-6 rounded pr-6 overflow-hidden"):
        with ui.grid(rows=1, columns=3).classes("h-full w-full gap-5 pb-4"):
            # Erste Spalte -> Age
            with ui.element("container_output_age").classes(
                    "rounded col-span-1 row-start-1 col-start-1 grid gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_age
                with ui.element("output_age").classes(
                        "flex flex-col justify-center items-start pl-10"):
                    age_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Age").classes("text-lg").style("color: white;")

            # Zweite Spalte -> Race
            with ui.element("container_output_race").classes(
                    "rounded col-span-1 row-start-1 col-start-2 grid gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_race
                with ui.element("output_race").classes(
                        "flex flex-col justify-center items-start pl-10"):
                    race_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Race").classes("text-lg").style("color: white;")

            # Dritte Spalte -> Gender
            with ui.element("container_output_gender").classes(
                    "rounded col-span-1 row-start-1 col-start-3 grid gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_gender
                with ui.element("output_gender").classes(
                        "flex flex-col justify-center items-start pl-10"):
                    gender_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Gender").classes("text-lg").style("color: white;")

    with ui.element("container_button").classes("col-span-4 row-start-6 flex overflow-hidden"):

        checkbox = ui.checkbox("Draw Landmarks").on_value_change(on_checkbox_change).classes("p-2 text-xl").style("color: white;")

        ui.button("Calculate", on_click=button_clicked).classes("btn btn-blue")

# A timer constantly updates the source of the image.
# Because data from same paths are cached by the browser,
# we must force an update by adding the current timestamp to the source.
ui.timer(interval=0.04, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))


async def disconnect() -> None:
    """Disconnect all clients from current running server."""
    for client_id in Client.instances:
        await core.sio.disconnect(client_id)


def handle_sigint(signum, frame) -> None:
    # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
    ui.timer(0.1, disconnect, once=True)
    # Delay the default handler to allow the disconnect to complete.
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    # This prevents ugly stack traces when auto-reloading on code change,
    # because otherwise disconnected clients try to reconnect to the newly started server.
    await disconnect()
    # Release the webcam hardware so it can be used by other applications again.
    video_capture.release()


app.on_shutdown(cleanup)

# We also need to disconnect clients when the app is stopped with Ctrl+C,
# because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
signal.signal(signal.SIGINT, handle_sigint)

ui.run(title="GUI")
