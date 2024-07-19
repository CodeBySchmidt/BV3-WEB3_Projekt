import cv2

# Video-Stream von der Kamera öffnen
cap = cv2.VideoCapture(0)

# Überprüfen, ob der Stream geöffnet werden konnte
if not cap.isOpened():
    print("Fehler beim Öffnen des Video-Streams")
    exit()

# Schleife zur kontinuierlichen Anzeige des Video-Streams
while True:
    # Frame vom Stream lesen
    ret, frame = cap.read()

    # Überprüfen, ob das Frame erfolgreich gelesen wurde
    if not ret:
        print("Fehler beim Lesen des Frames")
        break

    # Frame anzeigen
    cv2.imshow('Kamera Stream', frame)

    # Abbruchbedingung (wenn 'q' gedrückt wird)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stream und Fenster schließen
cap.release()
cv2.destroyAllWindows()
