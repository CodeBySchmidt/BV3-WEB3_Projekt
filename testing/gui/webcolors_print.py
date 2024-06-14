import webcolors


# CSS-Regeln initialisieren
css_colors = ""

# Schleife durch alle Farben und ihre HEX-Werte
for hex_value, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
    # CSS-Regel für jede Farbe hinzufügen
    css_colors += f".hex-{color_name.replace(' ', '-')} {{ background-color: {hex_value} !important; }}\n"

# Ausgabe der generierten CSS-Regeln
# print(css_colors)
