#VERSION 1 PHASER CON RED NEURONAL PARA EVITAR BALA HORIZONTAL
import pygame
import random
import os
import pandas as pd

from models import redNeuronal, regresion, decisionTree, knn

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Phaser")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []

#* variables para el modelo
modelo_actual = None
tipo_modelo = None
modelos = {
    'nn': redNeuronal.NeuralNetworkModel(),
    'dt': decisionTree.DecisionTreeModel(),
    'knn': knn.KNNModel(),
    'lr': regresion.LogisticRegressionModel()
}
#! mejora rendimiento
ULTIMA_PREDICCION = 0  # Tiempo de la última predicción
INTERVALO_PREDICCION = 0.2  # Segundos entre predicciones (5 veces/segundo)

# Función para seleccionar un modelo
def seleccionar_modelo(tipo):
    global modelo_actual, tipo_modelo
    if tipo not in modelos:
        print(f"Tipo de modelo {tipo} no reconocido")
        return False
        
    tipo_modelo = tipo
    modelo_actual = modelos[tipo]
    
    # Intentar cargar el modelo
    if modelo_actual.load():
        print(f"Modelo {modelo_actual.model_name} seleccionado y cargado")
        return True
    else:
        print(f"No se pudo cargar el modelo {modelo_actual.model_name}")
        return False

# Función para predecir salto con el modelo actual
def predecir_salto():
    global jugador, bala, velocidad_bala, modelo_actual
    
    if modelo_actual is None:
        print("No hay modelo seleccionado")
        return 0.0
    
    distancia = abs(jugador.x - bala.x)
    return modelo_actual.predict(velocidad_bala, distancia)

# Modo auto actualizado para usar el modelo actual
def modo_auto_prediccion():
    global salto, en_suelo, ULTIMA_PREDICCION
    tiempoActual = pygame.time.get_ticks() / 1000
    
    if (tiempoActual - ULTIMA_PREDICCION) > INTERVALO_PREDICCION:
        if modelo_actual is None or not modelo_actual.is_trained:
            print("Modelo no entrenado o no seleccionado")
            return
        
        if en_suelo:
            prob = predecir_salto()
            print(f"Probabilidad de salto: {prob}", end='\r')
            if prob > 0.49:  # Umbral ajustable
                salto = True
                en_suelo = False
        
        ULTIMA_PREDICCION = tiempoActual


# Cargar las imágenes
base_path = os.path.dirname(__file__)

jugador_frames = [
    pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_1.png')),
    pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_2.png')),
    pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_3.png')),
    pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_4.png'))
]

bala_img = pygame.image.load(os.path.join(base_path, 'assets/sprites/purple_ball.png'))
fondo_img = pygame.image.load(os.path.join(base_path, 'assets/game/fondo2.png'))
nave_img = pygame.image.load(os.path.join(base_path, 'assets/game/ufo.png'))
menu_img = pygame.image.load(os.path.join(base_path, 'assets/game/menu.png'))

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15  # Restablecer la velocidad de salto
            en_suelo = True

# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        reiniciar_juego()  # Terminar el juego y mostrar el menú

# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
    # Guardar velocidad de la bala, distancia al jugador y si saltó o no
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO)
    
    # Texto del menú principal
    titulo = fuente.render("Phaser y modelos", True, BLANCO)
    opcion_manual = fuente.render("M - Modo Manual (recolectar datos)", True, BLANCO)
    opcion_auto = fuente.render("A - Modo Automático (usar IA)", True, BLANCO)
    opcion_entrenar = fuente.render("T - Entrenar un modelo", True, BLANCO)
    opcion_salir = fuente.render("Q - Salir", True, BLANCO)
    
    # Mostrar opciones en pantalla
    pantalla.blit(titulo, (w // 4, 50))
    pantalla.blit(opcion_manual, (w // 4, 150))
    pantalla.blit(opcion_auto, (w // 4, 200))
    pantalla.blit(opcion_entrenar, (w // 4, 250))
    pantalla.blit(opcion_salir, (w // 4, 300))
    
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_m:  # Modo manual
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_a:  # Modo automático
                    mostrar_menu_seleccion_modelo()
                elif evento.key == pygame.K_t:  # Entrenar modelo
                    mostrar_menu_entrenamiento()
                elif evento.key == pygame.K_q:  # Salir
                    print("Juego terminado. Datos recopilados:", len(datos_modelo))
                    pygame.quit()
                    exit()

# Menú para seleccionar el modelo a usar en modo automático
def mostrar_menu_seleccion_modelo():
    global menu_activo, modo_auto
    
    pantalla.fill(NEGRO)
    titulo = fuente.render("Seleccione un modelo de IA:", True, BLANCO)
    opcion_nn = fuente.render("1 - Red Neuronal", True, BLANCO)
    opcion_dt = fuente.render("2 - Árbol de Decisión", True, BLANCO)
    opcion_knn = fuente.render("3 - K-Nearest Neighbors", True, BLANCO)
    opcion_lr = fuente.render("4 - Regresión Lineal", True, BLANCO)
    opcion_volver = fuente.render("5 - Volver al menú principal", True, BLANCO)
    
    pantalla.blit(titulo, (w // 4, 50))
    pantalla.blit(opcion_nn, (w // 4, 150))
    pantalla.blit(opcion_dt, (w // 4, 200))
    pantalla.blit(opcion_knn, (w // 4, 250))
    pantalla.blit(opcion_lr, (w // 4, 300))
    pantalla.blit(opcion_volver, (w // 4, 350))
    
    pygame.display.flip()
    
    seleccionando = True
    while seleccionando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:  # Red Neuronal
                    if seleccionar_modelo("nn"):
                        modo_auto = True
                        menu_activo = False
                        seleccionando = False
                    else:
                        mostrar_mensaje_error("Red Neuronal no entrenada")
                elif evento.key == pygame.K_2:  # Árbol de Decisión
                    if seleccionar_modelo("dt"):
                        modo_auto = True
                        menu_activo = False
                        seleccionando = False
                    else:
                        mostrar_mensaje_error("Árbol de Decisión no entrenado")
                elif evento.key == pygame.K_3:  # KNN
                    if seleccionar_modelo("knn"):
                        modo_auto = True
                        menu_activo = False
                        seleccionando = False
                    else:
                        mostrar_mensaje_error("KNN no entrenado")
                elif evento.key == pygame.K_4:  # Regresión Logística
                    if seleccionar_modelo("lr"):
                        modo_auto = True
                        menu_activo = False
                        seleccionando = False
                    else:
                        mostrar_mensaje_error("Regresión Logística no entrenada")
                elif evento.key == pygame.K_5:  # Volver
                    seleccionando = False

# Menú para entrenar un modelo
def mostrar_menu_entrenamiento():
    pantalla.fill(NEGRO)
    titulo = fuente.render("Seleccione un modelo para entrenar:", True, BLANCO)
    opcion_nn = fuente.render("1 - Red Neuronal", True, BLANCO)
    opcion_dt = fuente.render("2 - Árbol de Decisión", True, BLANCO)
    opcion_knn = fuente.render("3 - K-Nearest Neighbors", True, BLANCO)
    opcion_lr = fuente.render("4 - Regresión Logística", True, BLANCO)
    opcion_volver = fuente.render("5 - Volver al menú principal", True, BLANCO)
    
    pantalla.blit(titulo, (w // 4, 50))
    pantalla.blit(opcion_nn, (w // 4, 150))
    pantalla.blit(opcion_dt, (w // 4, 200))
    pantalla.blit(opcion_knn, (w // 4, 250))
    pantalla.blit(opcion_lr, (w // 4, 300))
    pantalla.blit(opcion_volver, (w // 4, 350))
    
    pygame.display.flip()
    
    seleccionando = True
    while seleccionando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:  # Red Neuronal
                    entrenar_modelo_seleccionado("nn")
                    seleccionando = False
                elif evento.key == pygame.K_2:  # Árbol de Decisión
                    entrenar_modelo_seleccionado("dt")
                    seleccionando = False
                elif evento.key == pygame.K_3:  # KNN
                    entrenar_modelo_seleccionado("knn")
                    seleccionando = False
                elif evento.key == pygame.K_4:  # Regresión Logística
                    entrenar_modelo_seleccionado("lr")
                    seleccionando = False
                elif evento.key == pygame.K_5:  # Volver
                    seleccionando = False

# Función para mostrar mensajes de error
def mostrar_mensaje_error(mensaje):
    pantalla.fill(NEGRO)
    texto = fuente.render(mensaje, True, (255, 50, 50))  # Rojo claro
    texto_continuar = fuente.render("Presione cualquier tecla para continuar", True, BLANCO)
    
    pantalla.blit(texto, (w // 4, h // 2 - 30))
    pantalla.blit(texto_continuar, (w // 4, h // 2 + 30))
    
    pygame.display.flip()
    
    # Esperar a que el usuario presione una tecla
    esperando = True
    while esperando:
        for evento in pygame.event.get():
            if evento.type == pygame.KEYDOWN:
                esperando = False
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()

# Función para entrenar el modelo seleccionado
def entrenar_modelo_seleccionado(tipo):
    global modelo_actual, tipo_modelo, menu_activo
    
    if len(datos_modelo) < 10:
        mostrar_mensaje_error("Se necesitan más datos para entrenar. Juega más en modo manual.")
        return
    
    seleccionar_modelo(tipo)
    
    pantalla.fill(NEGRO)
    texto = fuente.render(f"Entrenando modelo {tipo}...", True, BLANCO)
    texto_espera = fuente.render("(Esto puede tardar unos momentos)", True, BLANCO)
    pantalla.blit(texto, (w // 4, h // 2 - 30))
    pantalla.blit(texto_espera, (w // 4, h // 2 + 10))
    pygame.display.flip()
    
    # Procesar eventos durante el entrenamiento para evitar congelamiento
    pygame.event.pump()
    
    # Entrenar el modelo
    resultado = modelo_actual.train(datos_modelo)
    
    # Mostrar resultado y volver al menú principal explícitamente
    if resultado:
        pantalla.fill(NEGRO)
        texto = fuente.render(f"Modelo {tipo} entrenado exitosamente!", True, (50, 255, 50))
        texto_continuar = fuente.render("Presione cualquier tecla para continuar", True, BLANCO)
        pantalla.blit(texto, (w // 4, h // 2 - 30))
        pantalla.blit(texto_continuar, (w // 4, h // 2 + 10))
        pygame.display.flip()
    else:
        pantalla.fill(NEGRO)
        texto = fuente.render(f"Error al entrenar el modelo {tipo}", True, (255, 50, 50))
        texto_continuar = fuente.render("Presione cualquier tecla para continuar", True, BLANCO)
        pantalla.blit(texto, (w // 4, h // 2 - 30))
        pantalla.blit(texto_continuar, (w // 4, h // 2 + 10))
        pygame.display.flip()
    
    # Esperar a que el usuario presione una tecla
    esperando = True
    while esperando:
        for evento in pygame.event.get():
            if evento.type == pygame.KEYDOWN:
                esperando = False
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    # Asegurarse de volver al menú principal
    menu_activo = True
    mostrar_menu()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True
    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    global salto, en_suelo, bala_disparada, modo_auto, modelo_actual, datos_modelo, menu_activo
    
    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()
            
            # Modo automático
            if modo_auto:
                if modelo_actual and modelo_actual.is_trained:
                    modo_auto_prediccion()
                    if salto:
                        manejar_salto()
                else:
                    print("Entrena primero el modelo jugando en manual")
                    modo_auto = False
                    menu_activo = True
                    mostrar_menu()
            
            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
