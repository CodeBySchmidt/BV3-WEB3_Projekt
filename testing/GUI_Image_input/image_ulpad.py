import os
from nicegui import ui
import random

# Globale Referenz auf das Bild-Widget
image_display = None


def handle_upload(event):
    global image_display

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
        image_display.set_source(f'{save_path}')  # Ersetze die Bildquelle
    else:
        # Zeige das Bild zum ersten Mal an
        image_display = ui.image(f'{save_path}')


# Upload-Komponente
ui.upload(on_upload=handle_upload, label="Bild hochladen")

ui.run()
