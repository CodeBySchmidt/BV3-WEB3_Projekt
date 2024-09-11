import cv2
import numpy as np

# Bild einlesen
image = cv2.imread("../Utils/img/glasses/1.jpg")
# image = cv2.imread("cropped_image.jpg")
# Höhe und Breite des Bildes ermitteln
height, width = image.shape[:2]

# Definieren Sie die Höhe des oberen Teils (z.B. obere 50% des Bildes)
upper_part_height = int(height * 0.25)

# Ausschneiden des oberen Teils
upper_part = image[:upper_part_height, :]

# Anzeigen des oberen Teils
cv2.imshow('Upper Part of the Image', upper_part)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Falls das Bild in Farbe ist, den Median- und Mittelwert für jeden Kanal berechnen
if len(image.shape) == 3:  # Farb-Bild
    median_b = np.median(image[:,:,0].flatten())
    median_g = np.median(image[:,:,1].flatten())
    median_r = np.median(image[:,:,2].flatten())
    median_color = (median_b, median_g, median_r)

    mean_b = np.mean(image[:,:,0].flatten())
    mean_g = np.mean(image[:,:,1].flatten())
    mean_r = np.mean(image[:,:,2].flatten())
    mean_color = (mean_b, mean_g, mean_r)
else:  # Graustufen-Bild
    median_value = np.median(image.flatten())
    median_color = (median_value, median_value, median_value)

    mean_value = np.mean(image.flatten())
    mean_color = (mean_value, mean_value, mean_value)

# Ein neues Bild mit dem Median-Farbwert erstellen
median_image = np.full((height, width, 3), median_color, dtype=np.uint8)

# Ein neues Bild mit dem Mittelwert-Farbwert erstellen
mean_image = np.full((height, width, 3), mean_color, dtype=np.uint8)

print(f"Der Median-Farbwert des Bildes ist: {median_color}")
print(f"Der Mittelwert-Farbwert des Bildes ist: {mean_color}")

# Bild anzeigen
cv2.imshow("Image", image)
cv2.imshow('Median Color', median_image)
cv2.imshow('Mean Color', mean_image)
cv2.imshow('Upper Part of the Image', upper_part)
cv2.waitKey(0)
cv2.destroyAllWindows()
