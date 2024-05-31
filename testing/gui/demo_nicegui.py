from nicegui import ui
import webcolors

# You can also apply Tailwind CSS utility classes with the classes method ---> https://tailwindcss.com/docs/height (
# Height class) https://tailwindcss.com/docs/ ---> Für die Tailwind Classen Übersicht
# https://htmlcolorcodes.com/color-names/ ---> Hier eine Liste mit den Namen der Farben, zur Übersicht


# Definiere benutzerdefinierte CSS-Klassen für die Farben
ui.add_head_html('''
<style>
    .bg-red { background-color: red; }
    .bg-green { background-color: green; }
    .bg-blue { background-color: blue; }
    .bg-yellow { background-color: yellow; }
    .bg-purple { background-color: purple; }
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

# Erstelle das Grid mit den farbigen Labels
with ui.grid(columns=16).classes('w-full gap-2'):

    ui.label('full').classes('col-span-full border p-1 bg-red h-36')

    with ui.card().classes('col-span-8 border p-1 bg-green'):
        ui.label('Card content')
        ui.button('Add label', on_click=lambda: ui.label('Click!'))
        ui.timer(1.0, lambda: ui.label('Tick!'), once=True)

    ui.label('8').classes('col-span-8 border p-1 bg-blue h-36')

    ui.label('12').classes('col-span-12 border p-1 bg-yellow')

    ui.label('4').classes('col-span-4 border p-1 bg-purple')

    ui.label('15').classes('col-[span_15] border p-1 bg-orange')

    ui.label('1').classes('col-span-1 border p-1 bg-cyan')

    # Füge zusätzliche Labels mit Farben aus webcolors.CSS3_HEX_TO_NAMES hinzu, zum testen
    ui.label('DarkSlateGray').classes('col-span-full border p-1 h-10 bg-darkslategray')

    ui.label('Silver').classes('col-span-full border p-1 h-14 bg-silver')

    ui.label('Gainsboro').classes('col-span-full border p-1 h-16 bg-gainsboro')


# Starte die niceGUI-Anwendung und setze den Titel des Tabs
ui.run(title='Grid mit Farben')
