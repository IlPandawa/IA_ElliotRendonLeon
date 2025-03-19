import numpy as np
import cv2 as cv
import math 
import os

base_path = os.path.dirname(__file__)

rostro = cv.CascadeClassifier(os.path.join(base_path,'haarcascade_frontalface_alt.xml'))
cap = cv.VideoCapture(0)
i = 0  
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
       #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
       frame2 = frame[ y:y+h, x:x+w]
       #frame3 = frame[x+30:x+w-30, y+30:y+h-30]
       frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
       
        
       ruta = os.path.join('actividad4_Haarcascade\elliot', f'elliot{i}.jpg')
       if i%30 == 0:
        cv.imwrite(ruta, frame)
       cv.imshow('rostror', frame2)
    cv.imshow('rostros', frame)
    i = i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()