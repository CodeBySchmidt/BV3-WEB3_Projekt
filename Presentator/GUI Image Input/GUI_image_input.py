import random
import asyncio
import os

from nicegui import Client, app, core, ui, run
from facial_features_interface import *


# Globale Referenz auf das Bild-Widget
save_path = ""
image_display = None
hair_hex, hair_color_name, hair_typ = None, None, None
eye_color_hex, eye_color_name = None, None
glasses_result = None
facial_hair_result = None
age_gender_race_result = None


async def image_processing():
    global hair_hex, hair_color_name, hair_typ
    global eye_color_hex, eye_color_name
    global glasses_result

    await asyncio.sleep(0.1)  # Async sleep anstelle von blockierendem sleep
    ui.notify('Image processing task has started...')

    hair_hex, hair_color_name, hair_typ = await run.cpu_bound(hair_color, save_path)  # await hair_color(save_path)

    eye_color_hex, eye_color_name = await run.cpu_bound(eye_color, save_path)  # await eye_color(save_path)
    glasses_result = await run.cpu_bound(glasses, save_path)  # await glasses(save_path)

    # Labels aktualisieren, um den neuen Wert anzuzeigen
    ui.timer(1.0, lambda: hair_typ_label.set_text(hair_typ))
    ui.timer(1.0, lambda: eye_color_label.set_text(eye_color_hex.upper()))
    ui.timer(1.0, lambda: glasses_label.set_text(glasses_result))
    ui.timer(1.0, lambda: hair_color_label.set_text(hair_hex.upper()))

    # Hintergrundfarben der Labels aktualisieren
    output_1_color.style(f"background-color: {eye_color_hex} !important;")
    ui.timer(1.0, lambda: output_1_color.set_text(eye_color_hex.upper()))

    # Hair Color
    output_3_color.style(f"background-color: {hair_hex} !important;")
    ui.timer(1.0, lambda: output_3_color.set_text(hair_hex.upper()))

    ui.notify('Image processing task is finished')


async def neural_networks():
    global facial_hair_result, age_gender_race_result
    await asyncio.sleep(0.1)
    ui.notify('Neural network task has started...')
    facial_hair_result = await run.cpu_bound(facial_hair)  # facial_hair()
    age_gender_race_result = await run.cpu_bound(gender_age_race, save_path)

    # Labels aktualisieren, um den neuen Wert anzuzeigen
    ui.timer(1.0, lambda: facial_hair_label.set_text(facial_hair_result))
    ui.timer(1.0, lambda: age_label.set_text(age_gender_race_result[0]))
    ui.timer(1.0, lambda: gender_label.set_text(age_gender_race_result[1]))
    ui.timer(1.0, lambda: race_label.set_text(age_gender_race_result[2]))
    ui.notify('Neural network task is finished')

async def button_clicked():
    ui.notify('Calculating has started!')

    await image_processing()
    await neural_networks()

    await asyncio.sleep(1)
    ui.notify('All inputs are updated now')

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


def handle_upload(event):
    global image_display
    global save_path

    # Erstelle einen Ordner, wenn dieser noch nicht existiert
    save_folder = 'uploads/'
    os.makedirs(save_folder, exist_ok=True)

    # Bestimme den Dateinamen (Zufallswert, um das Caching zu umgehen)
    random_param = random.randint(1, 10000)
    new_filename = f'current_image_{random_param}.png'
    save_path = os.path.join(save_folder, new_filename)

    # Speichere die Datei mit dem neuen Namen
    with open(save_path, 'wb') as f:
        f.write(event.content.read())

    # Wenn bereits ein Bild angezeigt wird, ersetze es
    if image_display is not None:
        with ui.element("display_container"):
            image_display.set_source(f'{save_path}')  # Ersetze die Bildquelle
            print(save_path)
    else:
        # Zeige das Bild zum ersten Mal an
        ui.label("Your Picture").classes("text-3xl p-3").style("color: white; text-align: center;")
        with ui.element("display_container").classes("pb-12").style('display: flex; justify-content: center;'):
            # ui.label("Our Picture").classes("text-3xl").style("color: white;")
            image_display = ui.image(f'{save_path}').style("object-fit: contain; height: 50%; width: 50%;")
            print(save_path)


ui.query("body").classes("primary")

ui.image("Logo.svg").classes("w-full")

with ui.grid(columns=2).classes("w-full h-full pb-3"):

    with ui.element("OUTPUTS").classes("w-full h-full"):

        with ui.grid(rows=6, columns=2).classes("h-full w-full gap-5"):

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
            with ui.element("container_output_2").classes("rounded col-span-2 row-start-2 grid gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_2 links
                with ui.element("output_2").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
                    hair_typ_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Hair Type").classes("text-lg").style("color: white;")

            # Dritte Zeile -> Eye Color
            with ui.element("container_output_3").classes("rounded col-span-2 row-start-3 grid grid-cols-2 gap-2 secondary overflow-hidden drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_1 links
                with ui.element("output_1").classes("col-start-1 col-span-1 row-start-1 row-span-1 flex flex-col justify-center items-start pl-10"):
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

            # Vierte Zeile # output_4 links -> Facial Hair
            with ui.element("output_4").classes(
                    "col-start-1 col-span-1 row-start-5 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                age_label = ui.label("None").classes("text-3xl").style("color: white;")
                ui.label("Age").classes("text-lg").style("color: white;")

            # Zweite Zeile # output_5 rechts -> Glasses
            with ui.element("output_5").classes(
                    "col-start-2 col-span-1 row-start-5 row-span-1 rounded secondary overflow-hidden flex flex-col justify-center items-start pl-10 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_gender
                gender_label = ui.label("None").classes("text-3xl").style("color: white;")
                ui.label("Gender").classes("text-lg").style("color: white;")

            # Dritte Spalte -> Race
            with ui.element("container_output_race").classes("rounded col-span-2 row-start-6 grid gap-2 secondary overflow-hidden py-6 drop-shadow-[10px_12px_3px_rgba(0,0,0,0.25)]"):
                # output_race
                with ui.element("output_gender").classes("flex flex-col justify-center items-start pl-10"):
                    race_label = ui.label("None").classes("text-3xl").style("color: white;")
                    ui.label("Race").classes("text-lg").style("color: white;")

            ui.button("Calculate", on_click=button_clicked).classes("btn btn-blue")

    with ui.element("container_upload").classes("w-full").style("height: 800px"):
        ui.upload(on_upload=handle_upload, label="Bild hochladen").classes("w-full")


ui.run()
