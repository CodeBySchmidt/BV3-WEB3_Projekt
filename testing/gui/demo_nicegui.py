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

# Pfad zu deinem shape_predictor_68_face_landmarks.dat
predictor_path = "../Utils/shape_predictor_68_face_landmarks.dat"

# In case you don't have a webcam, this will provide a black placeholder image.
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')
# OpenCV is used to access the webcam.
video_capture = cv2.VideoCapture(0)

# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def convert(frame: np.ndarray) -> bytes:
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def detect_and_draw_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame


@app.get('/video/frame')
# Thanks to FastAPI's `app.get`` it is easy to create a web route which always provides the latest image from OpenCV.
async def grab_video_frame() -> Response:
    ret, frame = video_capture.read()
    if not ret:
        return placeholder

    frame_with_landmarks = detect_and_draw_landmarks(frame)
    jpeg = convert(frame_with_landmarks)
    return Response(content=jpeg, media_type='image/jpeg')

# Definiere benutzerdefinierte CSS-Klassen für die Farben
ui.add_head_html('''
<style>
    .primär { background-color: #42594C; }
    .sekundär { background-color: #9BF2BA; }
    .tertiär { background-color: #BF8D50; }
    .quartär { background-color: #F2BBB6; }
    .quintär { background-color: #0D0D0D; }
    .bg-orange { background-color: orange; }
    .bg-cyan { background-color: cyan; }
</style>
''')

# Erstelle benutzerdefinierte CSS-Klassen für die Farben aus webcolors.CSS3_HEX_TO_NAMES
css_colors = ""
for hex_value, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
    css_colors += f".bg-{color_name.replace(' ', '-')} {{ background-color: {hex_value}; }}\n"

ui.add_head_html(f'''
<style>
    {css_colors}
</style>
''')

ui.query('body').classes("primär")
# Erstelle das Grid mit den farbigen Labels
with (ui.grid(columns=16).classes('w-full gap-2')):
    with ui.card().classes('col-span-5 h-auto border p-1 sekundär'):
        ui.label('8')
        ui.button('Add label', on_click=lambda: ui.label('Click!'))
        ui.timer(1.0, lambda: ui.label('Tick!'), once=True)

    with ui.card().classes('col-span-11 h-auto border p-1 tertiär'):
        video_image = ui.interactive_image().classes('w-auto h-5/6')
        ui.label('Video image')

    ui.label('12').classes('col-span-12 border p-1 quartär')

    ui.label('4').classes('col-span-4 border p-1 quintär')

    ui.label('15').classes('col-[span_15] border p-1 bg-orange')

    ui.label('1').classes('col-span-1 border p-1 bg-cyan')

    # Füge zusätzliche Labels mit Farben aus webcolors.CSS3_HEX_TO_NAMES hinzu, zum testen
    ui.label('DarkSlateGray').classes('col-span-full border p-1 h-10 bg-darkslategray')

    ui.label('Silver').classes('col-span-full border p-1 h-14 bg-silver')

    ui.label('Gainsboro').classes('col-span-full border p-1 h-16 bg-gainsboro')

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
