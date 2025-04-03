## Análisis de Reconocimiento Facial utilizando mediapipe

El objetivo de la actividad es identificar una persona viva mediante reconocimiento facial con mediapipe, además de detectar sus emociones (tristeza, enojo, neutral, feliz).

Para ello podríamos tomar como referencia el análisis de los ojos, boca y cejas. Los parametrós para determinar que el reconocimiento mediante medipipe es a una persona viva y no una fotografía serían:

1. Análisis de movimiento: Se deben de rastrear el movimiento entre frames, puesto que una persona simepre hace ligeros movimientos, mientras que una foto parmanece estática
2. Detección de parpadeos: Se deben de detectar la transición entre los ojos abiertos y cerrados

Además para la detección se deben de evaluar la distancia o posición de los puntos faciales, en específico:
+ Boca: detectar si los puntos correspondientes a las comisuras de la boca estan hacia arriba o abajo dependiendo de la emoción, además de si esta abierta la boca o no
+ Cejas: si los puntos detectados estan arriba o abajo y además si están cerca (ceño fruncido)
+ Ojos: dependiendo de la distancia de apertura de los parpados puede indicar sorpresa o enojo



