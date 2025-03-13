import cv2
import numpy as np



# Ejemplo de como cargar una imagen con OpenCV
img = cv2.imread('actividad_OpenCV/natsuki.jpg', 1)
#cv2.imshow('salida', img)

# matriz de ceros
imgn = np.zeros(img.shape[:2], np.uint8)

#dividir los canales de color
b, g, r = cv2.split(img)
print(img.shape)

#combinar los canales de color
imgb = cv2.merge((b, imgn, imgn))
imgg = cv2.merge((imgn, g, imgn))
imgr = cv2.merge((imgn, imgn, r))
imgnn = cv2.merge([g, r , b])

# Quitar color
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Otras conversiones
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

""" cv2.imshow('salida', img)
cv2.imshow('salida2', img2)
cv2.imshow('salida3', img3)
cv2.imshow('salida4', img4) """

cv2.imshow('salida', b)
cv2.imshow('salida2', imgb)
cv2.imshow('salida3', imgg)
cv2.imshow('salida4', imgr)
cv2.imshow('salida5', imgn)
cv2.imshow('salida6', imgnn)

cv2.waitKey(0)
cv2.destroyAllWindows()