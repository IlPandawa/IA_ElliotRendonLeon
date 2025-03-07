import pygame
from queue import PriorityQueue # Para añadir a cola

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
        self.padre = None   #PAra guardar el nodo previo de donde viene
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
        self.color = ROJO

    def hacer_abierto(self):  # para pintar la casilla cuando entra a la lista abierta
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

# Función para obtener las casillas cercanas incluyendo las casillas en diagonal
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

    for df, dc, costo in movimientos:
        nf = nodo.fila + df
        nc = nodo.col + dc
        if 0 <= nf < filas and 0 <= nc < filas:
            vecino = grid[nf][nc]
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
    # implementación del algoritmo a*
    open_set = PriorityQueue()
    open_set.put((inicio.f, inicio))  # inicia con el nodo de inicio
    visitados = set()
    contador = 1
    
    # inicializa nodo inicial
    inicio.actualizar_costos(0, heuristica(inicio.get_pos(), fin.get_pos()), None, contador)
    
    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        nodo_actual = open_set.get()[1]  # obtiene nodo con menor f
        
        if nodo_actual == fin:
            reconstruir_camino(fin, ventana, grid, len(grid), ancho)
            return True
        
        for vecino, costo in obtener_cercanas(nodo_actual, grid):
            nuevo_g = nodo_actual.g + costo
            if nuevo_g < vecino.g:  # encuentra mejor camino
                contador += 1
                nuevo_h = heuristica(vecino.get_pos(), fin.get_pos())
                vecino.actualizar_costos(nuevo_g, nuevo_h, nodo_actual, contador)
                if vecino not in visitados:
                    open_set.put((vecino.f, vecino))
                    visitados.add(vecino)
                vecino.hacer_abierto()
        
        if nodo_actual != inicio:
            nodo_actual.hacer_cerrado()
        
        dibujar(ventana, grid, len(grid), ancho)
        pygame.time.delay(300)  # pausa para visualización
    
    return False  # no hay camino

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
    FILAS = 10
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