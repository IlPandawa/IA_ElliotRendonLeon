import pygame
from queue import PriorityQueue 

# Configuraciones iniciales
ANCHO_VENTANA = 600
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)

AZUL = (135, 206, 250)

pygame.font.init()  # para poder escribir en el tablero
FUENTE = pygame.font.Font(None, 18)  # Fuente 

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

        # VARIABLES PARA A*
        self.g = float("inf")  #Costo par amoverse
        self.h = 0      #Variable para la distancia hacia la salida
        self.f = float("inf")  # para la suma g+h y determinar a donde debe moverse
        self.padre = None   #PAra guardar el nodo de donde viene
        self.contador = 0   #Variable para el orden de movimientos


    # Función para actualizar los atributos
    def actualizar_costos(self, g, h, padre, contador):
        self.g = g  # nuevo costo
        self.h = h  # nueva heurística
        self.f = g + h  # nuevo valor total
        self.padre = padre  # nodo del que viene
        self.contador = contador  # contador de movimientos

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = BLANCO
            # tambien hay que restablecer el nodo a neutro
            self.g = float("inf")
            self.h = 0
            self.f = float("inf")
            self.padre = None
            self.contador = 0

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_cerrado(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = ROJO

    # para pintar la casilla cuando entra a la lista abierta
    def hacer_abierto(self):
        if not self.es_inicio() and not self.es_fin():
            self.color = AZUL

    def hacer_camino(self):  
        self.color = VERDE

    def __lt__(self, otro):  # Método para comparar nodos
        return self.f < otro.f

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        # pintar los valores de g h y f
        if self.g != float("inf") or self.h !=0:
            textoG = FUENTE.render(f"G:{self.g:.1f}", True, NEGRO)
            textoH = FUENTE.render(f"H:{self.h:.1f}", True, NEGRO)
            textoF = FUENTE.render(f"F:{self.f:.1f}", True, NEGRO)
            ventana.blit(textoG, (self.x + 2, self.y + 2))
            ventana.blit(textoH, (self.x + 2, self.y + 15))
            ventana.blit(textoF, (self.x + 2, self.y + 28))


# Función para obtener la distancia hacia la salida
def heuristica(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

# Función para obtener las casillas cercanas 
def obtener_cercanas(nodo, grid):
    filas = nodo.total_filas
    listaAbierta = []

    #Costo de los movimientos
    movimientos = [
        (-1, 0, 1), (1, 0, 1),          #movimientos verticales
        (0, -1, 1), (0, 1, 1),          #movimientos horizontal
        (-1, -1, 1.4), (-1, 1, 1.4),    #movimientos en diagonal
        (1, -1, 1.4), (1, 1, 1.4)
    ]

    for cambioFila, cambioColumna, costo in movimientos:
        nuevaFila = nodo.fila + cambioFila
        nuevaColumna = nodo.col + cambioColumna
        if 0 <= nuevaFila < filas and 0 <= nuevaColumna < filas:
            vecino = grid[nuevaFila][nuevaColumna]
            if not vecino.es_pared():  # ignora paredes como casilla valida
                listaAbierta.append((vecino, costo))
    return listaAbierta

def reconstruir_camino(nodo_actual, ventana, grid, filas, ancho):
    # traza el camino calculado por el algoritmo
    while nodo_actual.padre is not None:
        nodo_actual.hacer_camino()
        nodo_actual = nodo_actual.padre
        dibujar(ventana, grid, filas, ancho)  # actualiza el tablero

# ALGORITMO A*
def aEstrella(grid, inicio, fin, ventana, ancho):
    open_set = PriorityQueue()
    open_set.put((0, inicio))
    casillaAnterior = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())
    
    open_set_hash = {inicio}
    cerrados = set()

    # Actualizar valores iniciales del nodo
    inicio.actualizar_costos(
        g_score[inicio],
        heuristica(inicio.get_pos(), fin.get_pos()),
        None,
        1
    )

    while not open_set.empty():
        nodo_actual = open_set.get()[1]
        cerrados.add(nodo_actual)
        
        if nodo_actual == fin:
            # Reconstruir camino usando casillaAnterior
            while nodo_actual in casillaAnterior:
                nodo_actual = casillaAnterior[nodo_actual]
                nodo_actual.hacer_camino()
                dibujar(ventana, grid, len(grid), ancho)
            return True

        if nodo_actual in open_set_hash:
            open_set_hash.remove(nodo_actual)

        for vecino, costo in obtener_cercanas(nodo_actual, grid):
            if vecino in cerrados:
                continue

            tentative_g = g_score[nodo_actual] + costo
            
            if tentative_g < g_score[vecino]:
                casillaAnterior[vecino] = nodo_actual
                g_score[vecino] = tentative_g
                f_score[vecino] = tentative_g + heuristica(vecino.get_pos(), fin.get_pos())
                
                # Actualizar atributos del nodo
                vecino.actualizar_costos(
                    g_score[vecino],
                    heuristica(vecino.get_pos(), fin.get_pos()),
                    nodo_actual,
                    vecino.contador + 1
                )

                if vecino not in open_set_hash:
                    open_set.put((f_score[vecino], vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        nodo_actual.hacer_cerrado()
        dibujar(ventana, grid, len(grid), ancho)
        pygame.time.delay(250)

    return False

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    # Resetear nodos previos
                    for fila in grid:
                        for nodo in fila:
                            if nodo.color == VERDE or nodo.color == ROJO or nodo.color == AZUL:
                                nodo.restablecer()
                    aEstrella(grid, inicio, fin, ventana, ancho)

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)