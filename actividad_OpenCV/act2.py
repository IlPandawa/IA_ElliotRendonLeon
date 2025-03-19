import cv2 as cv
import numpy as np

# EJEMPLO PARA HACER SEGMENTACIÓN
img = cv.imread('actividad_OpenCV/manzana.jpg', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#Para trabajar con la segmentación definir un umbral bajo y un umbral alto
ub = np.array([0, 120, 40])
ua = np.array([10, 255, 255])

#Para abarcar ambos umbrales
ub1 = np.array([170, 40,40]) #Significado ([tono, saturación, brillo])
ua1 = np.array([180, 255, 255])

# Mascara con los valores del umbral bajo y umbral alto
#mask = cv.inRange(hsv, ub, ua)

#Mascaras con los umbrales distintos
mask1 = cv.inRange(hsv, ub, ua)
mask2 = cv.inRange(hsv, ub1, ua1)
mask = mask1 + mask2

#Imagen de resultado
res = cv.bitwise_and(img, img, mask=mask) # de la imagen original solamente va a dejar los valores que estén en la máscara


cv.imshow('hsv', hsv)
cv.imshow('mask', mask)
cv.imshow('res', res)

cv.waitKey(0)
cv.destroyAllWindows()
