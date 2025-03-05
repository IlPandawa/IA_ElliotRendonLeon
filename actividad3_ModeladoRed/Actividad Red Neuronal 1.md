## Elliot Rendón León

### Instrucciones
Modelar una red neuronal que pueda jugar al 5 en línea sin gravedad en un tablero de 20x20.
+ Definir el tipo de red neuronal y describir cada una de sus partes.
+ Definir los patrones a utilizar
+ Definir la función de activación que es necesaria para este problema
+ Definir el número máximo de entradas
+ ¿Qué valores a la salida de la red se podrían esperar?
+ ¿Cuáles son los valores máximos que puede tener el bias?

#### Propuesta de solución
Para este ejercicio se podría utilizar una red neuronal convolucional con una neurona por casilla que son 400 neuronas para este caso en especifico del tablero 20x20. Para las entradas de la red serían:
+ La matriz del tablero (0 si no tiene ficha puesta)
+ Fichas del rival (-1)
+ Fichas del jugador (1)

La conexión entre neuronas serviría para detectar lo patrones de las jugadas e identificar la posible mejor jugada. Utilizando la función sigmoide de activación y arrojar valores entre 0 y 1. 

En la matriz del tablero se tendrían 400 casillas en total, con un valor de 0, 1 o -1 dependiendo si ya contiene una ficha o no y si es del jugador o del adversario.
##### Valores esperados
Los valores esperados a la salida tendrían que ser números entre el 0 y el 1, donde mientras más acercado sea el valor a 1 es que será un mejor movimiento para poner una ficha.



