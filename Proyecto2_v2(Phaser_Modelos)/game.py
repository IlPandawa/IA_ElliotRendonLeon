import pygame
import random
import os
import time
from redNeuronal import (
    cargarRedSalto, cargarRedMovimiento, 
    entrenarRedSalto, entrenarRedMovimiento,
    pronosticarSalto, pronosticarMovimiento
)

from decisionTree import (
    cargar_modelo_salto_dt, cargar_modelo_movimiento_dt,
    entrenar_modelo_salto_dt, entrenar_modelo_movimiento_dt,
    predecir_salto_dt, predecir_movimiento_dt
)
from knn import (
    cargar_modelo_salto_knn, cargar_modelo_movimiento_knn,
    entrenar_modelo_salto_knn, entrenar_modelo_movimiento_knn,
    predecir_salto_knn, predecir_movimiento_knn
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

modelos_cargados = {
    "nn": {
        "salto": False,
        "movimiento": False
    },
    "dt": {
        "salto": False,
        "movimiento": False
    },
    "knn": {
        "salto": False,
        "movimiento": False
    }
}

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

# cargar modelos entrenados
modelos_cargados["nn"]["salto"] = cargarRedSalto()
modelos_cargados["nn"]["movimiento"] = cargarRedMovimiento()
modelos_cargados["dt"]["salto"] = cargar_modelo_salto_dt()
modelos_cargados["dt"]["movimiento"] = cargar_modelo_movimiento_dt()
modelos_cargados["knn"]["salto"] = cargar_modelo_salto_knn()
modelos_cargados["knn"]["movimiento"] = cargar_modelo_movimiento_knn()

# Cargar las imágenes
base_path = os.path.dirname(__file__)

try:
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
    """Maneja el movimiento del jugador en modo manual y registra la intención de movimiento"""
    global jugador, pos_actual
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        pos_actual = 0  # Intención: moverse a la izquierda
    elif keys[pygame.K_RIGHT]:
        pos_actual = 2  # Intención: moverse a la derecha
    else:
        pos_actual = 1  # Intención: quedarse quieto

    # Luego, movemos al jugador solo si está dentro de los límites
    if pos_actual == 0 and jugador.x > pos_x_min:
        jugador.x -= velocidad_x
    elif pos_actual == 2 and jugador.x < pos_x_max:
        jugador.x += velocidad_x

def aplicar_ia():
    global salto, en_suelo, jugador, pos_actual, ultima_prediccion_salto, ultima_prediccion_movimiento, tipo_modelo
    
    tiempo_actual = time.time()
    
    # Decidir si saltar (para evitar bala horizontal)
    if tiempo_actual - ultima_prediccion_salto > INTERVALO_PREDICCION:
        if en_suelo and bala_disparada:
            distancia = abs(jugador.x - bala.x)
            
            # Usar el modelo seleccionado para la predicción de salto
            if tipo_modelo == "nn":
                prob_salto = pronosticarSalto(velocidad_bala, distancia)
            elif tipo_modelo == "dt":
                prob_salto = predecir_salto_dt(velocidad_bala, distancia)
            elif tipo_modelo == "knn":
                prob_salto = predecir_salto_knn(velocidad_bala, distancia)
            else:
                prob_salto = 0.0
                
            print(f"Probabilidad de salto ({tipo_modelo}): {prob_salto:.2f}", end='\r')
            
            if prob_salto > 0.5:
                salto = True
                en_suelo = False
                print(f"IA ({tipo_modelo}): ¡Saltar!")
        
        ultima_prediccion_salto = tiempo_actual
    
    # Decidir movimiento horizontal (para evitar bala vertical)
    if tiempo_actual - ultima_prediccion_movimiento > INTERVALO_PREDICCION:
        if bala_vertical_disparada:
            # Usar el modelo seleccionado para la predicción de movimiento
            if tipo_modelo == "nn":
                accion = pronosticarMovimiento(
                    jugador.x, jugador.y,
                    bala_vertical.x, bala_vertical.y,
                    bala_vertical_disparada
                )
            elif tipo_modelo == "dt":
                accion = predecir_movimiento_dt(
                    jugador.x, jugador.y,
                    bala_vertical.x, bala_vertical.y,
                    bala_vertical_disparada
                )
            elif tipo_modelo == "knn":
                accion = predecir_movimiento_knn(
                    jugador.x, jugador.y,
                    bala_vertical.x, bala_vertical.y,
                    bala_vertical_disparada
                )
            else:
                accion = 1  # Por defecto, quedarse quieto
            
            # decisión de movimiento
            if accion == 0 and jugador.x > pos_x_min:  # Mover izquierda
                jugador.x -= velocidad_x
                pos_actual = 0
                print(f"IA ({tipo_modelo}): Mover izquierda")
            elif accion == 2 and jugador.x < pos_x_max:  # Mover derecha
                jugador.x += velocidad_x
                pos_actual = 2
                print(f"IA ({tipo_modelo}): Mover derecha")
            else:
                pos_actual = 1
                print(f"IA ({tipo_modelo}): Quieto")
        
        ultima_prediccion_movimiento = tiempo_actual


def guardar_datos():
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
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado.")
        print(f"Datos de salto: {len(datos_salto)} ejemplos")
        print(f"Datos de movimiento: {len(datos_movimiento)} ejemplos")
    else:
        print("Juego reanudado.")

def mostrar_menu():
    global menu_activo, modo_auto, tipo_modelo
    
    pantalla.fill(NEGRO)
    
    # Título
    titulo = fuente.render("Phaser Modelos", True, BLANCO)
    pantalla.blit(titulo, (w // 4, 30))
    
    # Opciones de menú
    y_pos = 100
    opciones = [
        {"texto": "Modo Manual (recolectar datos)", "tecla": "M", "color": BLANCO},
        {"texto": "Modo Automático - Red Neuronal", "tecla": "1", "color": BLANCO},
        {"texto": "Modo Automático - Árbol de Decisión", "tecla": "2", "color": BLANCO},
        {"texto": "Modo Automático - KNN", "tecla": "3", "color": BLANCO},
        {"texto": "Entrenar Modelos", "tecla": "T", "color": BLANCO},
        {"texto": "Salir", "tecla": "Q", "color": BLANCO}
    ]
    
    for opcion in opciones:
        texto = fuente.render(f"{opcion['tecla']} - {opcion['texto']}", True, opcion['color'])
        pantalla.blit(texto, (w // 4, y_pos))
        y_pos += 40
    
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
                elif evento.key == pygame.K_1:  # Red Neuronal
                    if modelos_cargados["nn"]["salto"] and modelos_cargados["nn"]["movimiento"]:
                        modo_auto = True
                        tipo_modelo = "nn"
                        menu_activo = False
                        print("Modo Automático (Red Neuronal) activado")
                    else:
                        print("¡Entrena primero el modelo de Red Neuronal!")
                elif evento.key == pygame.K_2:  # Árbol de Decisión
                    if modelos_cargados["dt"]["salto"] and modelos_cargados["dt"]["movimiento"]:
                        modo_auto = True
                        tipo_modelo = "dt"
                        menu_activo = False
                        print("Modo Automático (Árbol de Decisión) activado")
                    else:
                        print("¡Entrena primero el modelo de Árbol de Decisión!")
                elif evento.key == pygame.K_3:  # KNN
                    if modelos_cargados["knn"]["salto"] and modelos_cargados["knn"]["movimiento"]:
                        modo_auto = True
                        tipo_modelo = "knn"
                        menu_activo = False
                        print("Modo Automático (KNN) activado")
                    else:
                        print("¡Entrena primero el modelo KNN!")
                elif evento.key == pygame.K_t:
                    entrenar_modelos()
                elif evento.key == pygame.K_q:
                    print("Juego terminado.")
                    pygame.quit()
                    exit()

def entrenar_modelos():
    """Muestra un menú para seleccionar qué modelo entrenar"""
    global modelos_cargados
    
    pantalla.fill(NEGRO)
    titulo = fuente.render("Seleccione un modelo para entrenar:", True, BLANCO)
    
    opciones = [
        {"texto": "Red Neuronal", "tecla": "1", "tipo": "nn"},
        {"texto": "Árbol de Decisión", "tecla": "2", "tipo": "dt"},
        {"texto": "KNN", "tecla": "3", "tipo": "knn"},
        {"texto": "Todos los modelos", "tecla": "4", "tipo": "todos"},
        {"texto": "Volver al menú principal", "tecla": "5", "tipo": None}
    ]
    
    pantalla.blit(titulo, (w // 4, 50))
    
    y_pos = 120
    for opcion in opciones:
        texto = fuente.render(f"{opcion['tecla']} - {opcion['texto']}", True, BLANCO)
        pantalla.blit(texto, (w // 4, y_pos))
        y_pos += 40
    
    pygame.display.flip()
    
    seleccionando = True
    while seleccionando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:  # Red Neuronal
                    entrenar_modelo_especifico("nn")
                    seleccionando = False
                elif evento.key == pygame.K_2:  # Árbol de Decisión
                    entrenar_modelo_especifico("dt")
                    seleccionando = False
                elif evento.key == pygame.K_3:  # KNN
                    entrenar_modelo_especifico("knn")
                    seleccionando = False
                elif evento.key == pygame.K_4:  # Todos los modelos
                    entrenar_todos_modelos()
                    seleccionando = False
                elif evento.key == pygame.K_5:  # Volver
                    seleccionando = False
                    mostrar_menu()

def entrenar_modelo_especifico(tipo_modelo):
    global modelos_cargados
    
    pantalla.fill(NEGRO)
    texto = fuente.render(f"Entrenando modelo {tipo_modelo.upper()}...", True, BLANCO)
    pantalla.blit(texto, (w // 3, h // 2 - 30))
    pygame.display.flip()
    
    # Verificar datos suficientes
    if len(datos_salto) < 10:
        texto = fuente.render("No hay suficientes datos de salto", True, ROJO)
        pantalla.blit(texto, (w // 3, h // 2))
        pygame.display.flip()
        pygame.time.delay(2000)
        mostrar_menu()
        return
        
    if len(datos_movimiento) < 10:
        texto = fuente.render("No hay suficientes datos de movimiento", True, ROJO)
        pantalla.blit(texto, (w // 3, h // 2 + 30))
        pygame.display.flip()
        pygame.time.delay(2000)
        mostrar_menu()
        return
    
    # Entrenar modelo de salto
    exito_salto = False
    exito_movimiento = False
    
    if tipo_modelo == "nn":
        texto = fuente.render("Entrenando Red Neuronal (salto)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2))
        pygame.display.flip()
        exito_salto = entrenarRedSalto(datos_salto)
        
        texto = fuente.render("Entrenando Red Neuronal (movimiento)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2 + 30))
        pygame.display.flip()
        exito_movimiento = entrenarRedMovimiento(datos_movimiento)
        
    elif tipo_modelo == "dt":
        texto = fuente.render("Entrenando Árbol de Decisión (salto)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2))
        pygame.display.flip()
        exito_salto = entrenar_modelo_salto_dt(datos_salto)
        
        texto = fuente.render("Entrenando Árbol de Decisión (movimiento)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2 + 30))
        pygame.display.flip()
        exito_movimiento = entrenar_modelo_movimiento_dt(datos_movimiento)
        
    elif tipo_modelo == "knn":
        texto = fuente.render("Entrenando KNN (salto)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2))
        pygame.display.flip()
        exito_salto = entrenar_modelo_salto_knn(datos_salto)
        
        texto = fuente.render("Entrenando KNN (movimiento)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2 + 30))
        pygame.display.flip()
        exito_movimiento = entrenar_modelo_movimiento_knn(datos_movimiento)
    
    # Actualizar estado de modelos cargados
    if exito_salto:
        modelos_cargados[tipo_modelo]["salto"] = True
    if exito_movimiento:
        modelos_cargados[tipo_modelo]["movimiento"] = True
    
    # Mostrar resultado
    pantalla.fill(NEGRO)
    if exito_salto and exito_movimiento:
        texto = fuente.render(f"¡Modelo {tipo_modelo.upper()} entrenado con éxito!", True, VERDE)
    else:
        texto = fuente.render(f"Entrenamiento parcial de {tipo_modelo.upper()}", True, ROJO)
    
    pantalla.blit(texto, (w // 3, h // 2))
    pygame.display.flip()
    pygame.time.delay(2000)
    
    # Volver al menú principal
    mostrar_menu()

def entrenar_todos_modelos():
    """Entrena todos los modelos disponibles"""
    global modelos_cargados
    
    pantalla.fill(NEGRO)
    texto = fuente.render("Entrenando todos los modelos...", True, BLANCO)
    pantalla.blit(texto, (w // 3, h // 2 - 60))
    pygame.display.flip()
    
    # Verificar datos suficientes
    if len(datos_salto) < 10 or len(datos_movimiento) < 10:
        texto = fuente.render("No hay suficientes datos (mínimo 10)", True, ROJO)
        pantalla.blit(texto, (w // 3, h // 2))
        pygame.display.flip()
        pygame.time.delay(2000)
        mostrar_menu()
        return
    
    # Entrenar todos los modelos de salto
    y_offset = -30
    for tipo, nombre in [("nn", "Red Neuronal"), ("dt", "Árbol de Decisión"), ("knn", "KNN")]:
        y_offset += 60
        
        # Entrenar modelo de salto
        texto = fuente.render(f"Entrenando {nombre} (salto)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2 - 30 + y_offset))
        pygame.display.flip()
        
        if tipo == "nn":
            if entrenarRedSalto(datos_salto):
                modelos_cargados[tipo]["salto"] = True
        elif tipo == "dt":
            if entrenar_modelo_salto_dt(datos_salto):
                modelos_cargados[tipo]["salto"] = True
        elif tipo == "knn":
            if entrenar_modelo_salto_knn(datos_salto):
                modelos_cargados[tipo]["salto"] = True
        
        # Entrenar modelo de movimiento
        texto = fuente.render(f"Entrenando {nombre} (movimiento)...", True, BLANCO)
        pantalla.blit(texto, (w // 3, h // 2 + y_offset))
        pygame.display.flip()
        
        if tipo == "nn":
            if entrenarRedMovimiento(datos_movimiento):
                modelos_cargados[tipo]["movimiento"] = True
        elif tipo == "dt":
            if entrenar_modelo_movimiento_dt(datos_movimiento):
                modelos_cargados[tipo]["movimiento"] = True
        elif tipo == "knn":
            if entrenar_modelo_movimiento_knn(datos_movimiento):
                modelos_cargados[tipo]["movimiento"] = True
    
    # Mostrar resultados
    pantalla.fill(NEGRO)
    todos_entrenados = all(all(modelos_cargados[modelo].values()) for modelo in modelos_cargados)
    
    if todos_entrenados:
        texto = fuente.render("¡Todos los modelos entrenados con éxito!", True, VERDE)
    else:
        texto = fuente.render("Entrenamiento parcial completado", True, ROJO)
    
    pantalla.blit(texto, (w // 3, h // 2))
    pygame.display.flip()
    pygame.time.delay(2000)
    
    # Volver al menú principal
    mostrar_menu()

def reiniciar_juego():
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
    global salto, en_suelo, bala_disparada, jugador, tipo_modelo
    
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
                
                # Teclas para cambiar el modelo en tiempo real
                if evento.key == pygame.K_1 and modo_auto:
                    if modelos_cargados["nn"]["salto"] and modelos_cargados["nn"]["movimiento"]:
                        tipo_modelo = "nn"
                        print("Cambiado a Red Neuronal")
                    else:
                        print("Red Neuronal no entrenada")
                elif evento.key == pygame.K_2 and modo_auto:
                    if modelos_cargados["dt"]["salto"] and modelos_cargados["dt"]["movimiento"]:
                        tipo_modelo = "dt"
                        print("Cambiado a Árbol de Decisión")
                    else:
                        print("Árbol de Decisión no entrenado")
                elif evento.key == pygame.K_3 and modo_auto:
                    if modelos_cargados["knn"]["salto"] and modelos_cargados["knn"]["movimiento"]:
                        tipo_modelo = "knn"
                        print("Cambiado a KNN")
                    else:
                        print("KNN no entrenado")
        
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