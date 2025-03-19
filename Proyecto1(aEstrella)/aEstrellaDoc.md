### Reporte de implementación de algoritmo A*
Para la implementación del algoritmo A* se realizaron los siguientes cambios sobre el cascarón proporcionado:

#### Colores y fuente
Se agrego el color azul para pintar los nuevos nodos que van a entrar a la lista abierta, además se agrego la parte de las fuentes de `pygame.font` para poder dibujar en cada casilla los valores de g, h y f.
![[Pasted image 20250310100136.png]]

#### Modificación en clase nodo
Se agregaron las variables nuevas para poder guardar los valores de g, h, f y además las variables padre y contador para guardar el nodo de donde viene y un contador para almacenar el orden de los movimientos del algoritmo.
![[Pasted image 20250310101300.png]]

#### Método actualizar_costo y restablecer
Se agrego a la clase nodo este método para poder actualizar los valores de cada nodo mientras se realiza la búsqueda del mejor camino. Además se modifico el método restablecer que en el cascarón solo restablecía el color, ahora va a reiniciar la información de los nodos para una nueva búsqueda.
![[Pasted image 20250310101939.png]]

#### Modificación en dibujar
Se agrego la visualización de los valores de g, h y f.
![[Pasted image 20250310102206.png]]

#### Método heurística y obtener_cercanas
Se agrego el método para calcular la distancia entre el nodo donde se vaya a hacer el calculo y el nodo de salida, utilizando la distancia Manhattan. El método obtener cercanas obtiene las casillas cercanas al nodo donde se encuentre y los costos de los movimientos que haga.
![[Pasted image 20250311095632.png]]

#### Método Reconstruir camino
Pinta el camino óptimo una vez termina la búsqueda.
![[Pasted image 20250311095748.png]]

### **Algoritmo A****
El algoritmo inicia utilizando una cola para almacenar las casillas cercanas que se van a explorar desde la casilla inicial. Se utilizan diccionarios para guardar los costos de g, f, h.
Posteriormente en el bucle se va a extraer el nodo menor de la cola de prioridad, así va explorando los vecinos desde el nodo en el que se encuentre actualmente, además se hacen las validaciones correspondientes para ignorar las paredes y los nodos que ya hayan entrado a la lista cerrada. 
Finalmente, al termino de la búsqueda del algoritmo si llega al nodo final recorre el camino optimo trazando el camino de color verde mediante la `casillaAnterior`.

![[Pasted image 20250312213254.png]]


