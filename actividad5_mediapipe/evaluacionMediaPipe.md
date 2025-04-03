# Evaluación Redes Neuronales Mediapipe

## Nombre: **Elliot Rendón León**

### Instrucciones:
Modelar una red neuronal que pueda identificar emociones a través de los valores obtenidos de los landmarks que genera mediapipe

+ Definir el tipo de red neuronal y describir cada una de sus partes. 
+ Definir los patrones a utilizar
+ Definir que función de activación es necesaria para este problema
+ Definir el número máximo de entradas
+ ¿Que valores a la salida de la red se podrían esperar?
+ ¿Cúales son los valores máximos que puede tener el BIAS?
---
#### Respuestas
1. Podría utilizarse una red neuronal convolucional, que consta de capas de entrada, las capas ocultas y capas de salida
2. Los patrones a tomar en cuenta serían los landmarks de puntos clave que reflejan las emociones, como la boca, tanto su contorno interno como externo, además de las cejas y ojos. Por que de ellos se debe de detectar los patrones de distancia o posición de los landmarks.
    + Boca: detectar si los puntos correspondientes a las comisuras de la boca estan hacia arriba o abajo dependiendo de la emoción, además de si esta abierta la boca o no
    + Cejas: si los puntos detectados estan arriba o abajo y además si están cerca (ceño fruncido)
    + Ojos: la distancia entre los parpados
3. Para las funciones de activación sería una ReLU y una Softmax, ReLU para aprender de los cambios de expresión facial y la Softmax para convertir las salidas en probabilidades entre 0 y 1 para cada emoción, por ejemplo a la salida tendría que dar algo como:
    + Feliz: 0.85 en el caso de que este sonriendo y así para cada emoción
4. El número máximo de entradas sería una neurona por cada landmark, pero hay tomar en cuenta que se tienen que evaluar en 3 dimensiones (x, y, z) por lo que si mediapipe nos ofrece 468 landmarks habría +1,400 entradas, pero no es necesario utilizar todos los puntos si lo relevante para determinar emociones es la boca, ojos y cejas habría alrededor de 200-300 entradas
5. Como se utilizaría la función softmax arrojaría al final probabilidades entre 0 y 1 para cada emoción, podría arrojar algo como lo siguiente:
    + | Feliz | Triste | Enojado | Asombrado | Neutral |
    + | 0.85  | 0.02   | 0.07    | 0.03      | 0.03    |

    + Ahí habría que ajustar el umbral de cuanto estemos tomando para las probabilidades porque algunas expresiones se pueden parecer un poco y arrojar valores muy similares, como por ejemplo neutral y triste
6. Los valores del BIAS no tienen un máximo esos se van ajustando dependiendo del progreso del entrenamiento. *(Es como lo de la perilla de radio se va ajustando)*
