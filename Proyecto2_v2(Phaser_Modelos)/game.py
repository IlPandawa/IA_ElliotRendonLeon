import pygame
import random
import os
import time
from redNeuronal import (
    cargar_modelo_salto, cargar_modelo_movimiento, 
    entrenar_modelo_salto, entrenar_modelo_movimiento,
    predecir_salto, predecir_movimiento
)

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Phaser")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)

# Variables del jugador, balas, nave, fondo, etc.
jugador = None
bala = None
bala_vertical = None
fondo = None
nave = None

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

# Variables para el movimiento del jugador
pos_x_min = 20  # Límite mínimo de movimiento hacia la izquierda
pos_x_max = 100  # Límite máximo de movimiento hacia la derecha
velocidad_x = 5  # Velocidad de movimiento lateral
pos_actual = 1  # 0: izquierda, 1: quieto, 2: derecha

# Listas para guardar los datos para entrenamiento
datos_salto = []
datos_movimiento = []

# Variables para los modelos de IA
modelo_salto_cargado = False
modelo_movimiento_cargado = False

# Variables para control de predicciones
INTERVALO_PREDICCION = 0.2  # Segundos entre predicciones
ultima_prediccion_salto = 0
ultima_prediccion_movimiento = 0

# Variables para balas
velocidad_bala = -10  # Velocidad de la bala horizontal
bala_disparada = False
velocidad_bala_vertical = 4
bala_vertical_disparada = False
tiempo_ultimo_disparo = 0

# Intentar cargar modelos previamente entrenados
modelo_salto_cargado = cargar_modelo_salto()
modelo_movimiento_cargado = cargar_modelo_movimiento()

# Cargar las imágenes
base_path = os.path.dirname(__file__)

# Verificar existencia de imágenes y cargar usando rutas alternativas si es necesario
try:
    # Primero intenta usar el path original
    jugador_frames = [
        pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_1.png')),
        pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_2.png')),
        pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_3.png')),
        pygame.image.load(os.path.join(base_path, 'assets/sprites/mono_frame_4.png'))
    ]
    bala_img = pygame.image.load(os.path.join(base_path, 'assets/sprites/purple_ball.png'))
    fondo_img = pygame.image.load(os.path.join(base_path, 'assets/game/fondo2.png'))
    nave_img = pygame.image.load(os.path.join(base_path, 'assets/game/ufo.png'))
except FileNotFoundError:
    # Usar imágenes alternativas del directorio pygamesc
    try:
        jugador_frames = [
            pygame.transform.scale(pygame.image.load(r'pygamesc\assets\sprites\M\M0.png'), (44, 55)),
            pygame.transform.scale(pygame.image.load(r'pygamesc\assets\sprites\M\M1.png'), (44, 55)),
            pygame.transform.scale(pygame.image.load(r'pygamesc\assets\sprites\M\M2.png'), (44, 55)),
            pygame.transform.scale(pygame.image.load(r'pygamesc\assets\sprites\M\M3.png'), (44, 55))
        ]
        bala_img = pygame.transform.scale(pygame.image.load(r'pygamesc\assets\sprites\ball.png'), (44, 44))
        fondo_img = pygame.image.load(r'pygamesc\assets\game\fondo2.png')
        nave_img = pygame.image.load(r'pygamesc\assets\game\bowser.png')
    except FileNotFoundError:
        # Crear imágenes de respaldo como último recurso
        jugador_frames = [pygame.Surface((32, 48)) for _ in range(4)]
        for frame in jugador_frames:
            frame.fill(AZUL)
        bala_img = pygame.Surface((16, 16))
        bala_img.fill(ROJO)
        fondo_img = pygame.Surface((w, h))
        fondo_img.fill(NEGRO)
        nave_img = pygame.Surface((64, 64))
        nave_img.fill(VERDE)

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear los rectángulos del jugador, balas y nave
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
bala_vertical = pygame.Rect(random.randint(pos_x_min, pos_x_max), 0, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# ---------------- FUNCIONES DEL JUEGO ----------------

def disparar_bala():
    """Inicia el disparo de la bala horizontal"""
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-8, -3)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

def disparar_bala_vertical():
    """Inicia el disparo de la bala vertical"""
    global bala_vertical_disparada, bala_vertical
    if not bala_vertical_disparada:
        bala_vertical.x = random.randint(pos_x_min, pos_x_max)  # Posición X aleatoria
        bala_vertical.y = 0  # Iniciar desde la parte superior
        bala_vertical_disparada = True

def reset_bala():
    """Reinicia la posición de la bala horizontal"""
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

def reset_bala_vertical():
    """Reinicia la posición de la bala vertical"""
    global bala_vertical, bala_vertical_disparada
    bala_vertical.y = 0
    bala_vertical_disparada = False

def manejar_salto():
    """Maneja la física del salto del jugador"""
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

def mover_jugador_manual():
    """Maneja el movimiento del jugador en modo manual"""
    global jugador, pos_actual
    keys = pygame.key.get_pressed()

    # Movimiento horizontal
    if keys[pygame.K_LEFT] and jugador.x > pos_x_min:
        jugador.x -= velocidad_x
        pos_actual = 0
    elif keys[pygame.K_RIGHT] and jugador.x < pos_x_max:
        jugador.x += velocidad_x
        pos_actual = 2
    else:
        pos_actual = 1

def aplicar_ia():
    """Aplica decisiones de la IA para el movimiento y salto del jugador"""
    global salto, en_suelo, jugador, pos_actual, ultima_prediccion_salto, ultima_prediccion_movimiento
    
    tiempo_actual = time.time()
    
    # Decidir si saltar (para evitar bala horizontal)
    if tiempo_actual - ultima_prediccion_salto > INTERVALO_PREDICCION:
        if en_suelo and bala_disparada:
            distancia = abs(jugador.x - bala.x)
            prob_salto = predecir_salto(velocidad_bala, distancia)
            print(f"Probabilidad de salto: {prob_salto:.2f}", end='\r')
            
            if prob_salto > 0.5:
                salto = True
                en_suelo = False
                print("IA: ¡Saltar!")
        
        ultima_prediccion_salto = tiempo_actual
    
    # Decidir movimiento horizontal (para evitar bala vertical)
    if tiempo_actual - ultima_prediccion_movimiento > INTERVALO_PREDICCION:
        if bala_vertical_disparada:
            accion = predecir_movimiento(
                jugador.x, jugador.y,
                bala_vertical.x, bala_vertical.y,
                bala_vertical_disparada
            )
            
            # Aplicar la decisión de movimiento
            if accion == 0 and jugador.x > pos_x_min:  # Mover izquierda
                jugador.x -= velocidad_x
                pos_actual = 0
                print("IA: Mover izquierda")
            elif accion == 2 and jugador.x < pos_x_max:  # Mover derecha
                jugador.x += velocidad_x
                pos_actual = 2
                print("IA: Mover derecha")
            else:
                pos_actual = 1
                print("IA: Quieto")
        
        ultima_prediccion_movimiento = tiempo_actual

def guardar_datos():
    """Guarda los datos para el entrenamiento de los modelos"""
    global jugador, bala, bala_vertical, velocidad_bala, salto, pos_actual
    
    # Datos para el modelo de salto
    if bala_disparada:
        distancia = abs(jugador.x - bala.x)
        salto_hecho = 1 if salto else 0
        datos_salto.append((velocidad_bala, distancia, salto_hecho))
    
    # Datos para el modelo de movimiento
    if bala_vertical_disparada:
        distancia_x = jugador.x - bala_vertical.x
        distancia_y = jugador.y - bala_vertical.y
        
        datos_movimiento.append((
            jugador.x, jugador.y,
            bala_vertical.x, bala_vertical.y,
            distancia_x, distancia_y,
            1 if bala_vertical_disparada else 0,
            pos_actual  # 0: izquierda, 1: quieto, 2: derecha
        ))

def update():
    """Actualiza el estado del juego y dibuja en pantalla"""
    global bala, bala_vertical, current_frame, frame_count, fondo_x1, fondo_x2
    global tiempo_ultimo_disparo, bala_vertical_disparada
    
    # Lógica para disparar la bala vertical aleatoriamente
    tiempo_actual = pygame.time.get_ticks()
    if not bala_vertical_disparada and tiempo_actual - tiempo_ultimo_disparo > random.randint(2000, 5000):
        disparar_bala_vertical()
        tiempo_ultimo_disparo = tiempo_actual

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    if fondo_x1 <= -w:
        fondo_x1 = w
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

    # Mover y dibujar la bala horizontal
    if bala_disparada:
        bala.x += velocidad_bala
        pantalla.blit(bala_img, (bala.x, bala.y))

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    # Mover y dibujar la bala vertical
    if bala_vertical_disparada:
        bala_vertical.y += velocidad_bala_vertical
        pantalla.blit(bala_img, (bala_vertical.x, bala_vertical.y))
        
        # Si la bala llega al suelo, resetearla
        if bala_vertical.y > h:
            reset_bala_vertical()

    # Colisión entre las balas y el jugador
    if jugador.colliderect(bala) or jugador.colliderect(bala_vertical):
        print("Colisión detectada!")
        reiniciar_juego()

def pausa_juego():
    """Pausar o reanudar el juego"""
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
        print(f"Datos de salto: {len(datos_salto)} ejemplos")
        print(f"Datos de movimiento: {len(datos_movimiento)} ejemplos")
    else:
        print("Juego reanudado.")

def mostrar_menu():
    """Muestra el menú principal del juego"""
    global menu_activo, modo_auto
    
    pantalla.fill(NEGRO)
    
    # Título
    titulo = fuente.render("Phaser", True, BLANCO)
    pantalla.blit(titulo, (w // 4, h // 4))
    
    # Opciones de menú
    opciones = [
        "Presiona 'M' para Modo Manual",
        "Presiona 'A' para Modo Automático (IA)",
        "Presiona 'T' para Entrenar Modelos",
        "Presiona 'Q' para Salir"
    ]
    
    for i, opcion in enumerate(opciones):
        texto = fuente.render(opcion, True, BLANCO)
        pantalla.blit(texto, (w // 4, h // 3 + i * 30))
    
    # Estado de modelos
    estado_salto = "Cargado" if modelo_salto_cargado else "No entrenado"
    estado_mov = "Cargado" if modelo_movimiento_cargado else "No entrenado"
    
    estado = fuente.render(f"Modelo Salto: {estado_salto} | Modelo Movimiento: {estado_mov}", True, VERDE if modelo_salto_cargado and modelo_movimiento_cargado else ROJO)
    pantalla.blit(estado, (w // 4, h // 3 + len(opciones) * 30 + 20))
    
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                    print("Modo Manual activado")
                elif evento.key == pygame.K_a:
                    if modelo_salto_cargado and modelo_movimiento_cargado:
                        modo_auto = True
                        menu_activo = False
                        print("Modo Automático (IA) activado")
                    else:
                        print("¡Entrena los modelos primero!")
                elif evento.key == pygame.K_t:
                    entrenar_modelos()
                elif evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()

def entrenar_modelos():
    """Entrena los modelos con los datos recopilados"""
    global modelo_salto_cargado, modelo_movimiento_cargado
    
    pantalla.fill(NEGRO)
    texto = fuente.render("Entrenando modelos...", True, BLANCO)
    pantalla.blit(texto, (w // 3, h // 2))
    pygame.display.flip()
    
    # Entrenar modelo de salto
    if len(datos_salto) >= 10:
        if entrenar_modelo_salto(datos_salto):
            modelo_salto_cargado = True
    else:
        texto = fuente.render("No hay suficientes datos de salto", True, ROJO)
        pantalla.blit(texto, (w // 3, h // 2 + 30))
        pygame.display.flip()
        pygame.time.delay(2000)
    
    # Entrenar modelo de movimiento
    if len(datos_movimiento) >= 10:
        if entrenar_modelo_movimiento(datos_movimiento):
            modelo_movimiento_cargado = True
    else:
        texto = fuente.render("No hay suficientes datos de movimiento", True, ROJO)
        pantalla.blit(texto, (w // 3, h // 2 + 60))
        pygame.display.flip()
        pygame.time.delay(2000)
    
    # Mostrar resultados
    pantalla.fill(NEGRO)
    
    if modelo_salto_cargado and modelo_movimiento_cargado:
        texto = fuente.render("¡Modelos entrenados con éxito!", True, VERDE)
        #delay
        pygame.time.delay(1000)
        mostrar_menu()
    else:
        texto = fuente.render("Entrenamiento incompleto", True, ROJO)
    
    pantalla.blit(texto, (w // 3, h // 2))
    pygame.display.flip()
    pygame.time.delay(2000)

def reiniciar_juego():
    """Reinicia el juego tras una colisión"""
    global menu_activo, jugador, bala, bala_vertical, nave
    global bala_disparada, bala_vertical_disparada, salto, en_suelo
    
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala horizontal
    bala_vertical.y = 0  # Reiniciar posición de la bala vertical
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    bala_vertical_disparada = False
    salto = False
    en_suelo = True
    
    # Mostrar estadísticas
    print(f"Datos acumulados - Salto: {len(datos_salto)}, Movimiento: {len(datos_movimiento)}")
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    """Función principal del juego"""
    global salto, en_suelo, bala_disparada, jugador
    
    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True
    
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:
                    pausa_juego()
                if evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()
                if evento.key == pygame.K_t:
                    entrenar_modelos()
        
        if not pausa:
            # Modo manual
            if not modo_auto:
                mover_jugador_manual()
                if salto:
                    manejar_salto()
                guardar_datos()
            # Modo automático (IA)
            else:
                aplicar_ia()
                if salto:
                    manejar_salto()
            
            # Disparar balas si no están ya en movimiento
            if not bala_disparada:
                disparar_bala()
            
            # Actualizar el juego
            update()
        
        # Actualizar pantalla
        pygame.display.flip()
        reloj.tick(30)  # 30 FPS
    
    pygame.quit()

if __name__ == "__main__":
    main()