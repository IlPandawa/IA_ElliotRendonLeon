# Documentación Phaser con modelos

## Introducción
Este proyecto implementa un juego estilo Phaser donde un personaje debe evadir proyectiles horizontales y verticales, utilizando diferentes modelos para controlar el movimiento del personaje en modo automático. Se han implementado tres tipos de modelos de aprendizaje automático que pueden entrenarse con datos recopilados durante el juego en modo manual:

1. Red Neuronal
2. Árbol de Decisión
3. K-Nearest Neighbors (KNN)

## Estructura General del Proyecto
El proyecto está organizado en varios archivos:

- `game.py`: Contiene la lógica principal del juego y la interfaz.
- `redNeuronal.py`: Implementación del modelo de Red Neuronal.
- `decisionTree.py`: Implementación del modelo de Árbol de Decisión.
- `knn.py`: Implementación del modelo de K-Nearest Neighbors.

Los modelos entrenados se guardan en la carpeta modelosGuardados.

## Implementación del Juego Principal (game.py)
#### Inicialización y Configuración
El juego utiliza pygame para renderizar gráficos y manejar eventos. Se establecen dimensiones de pantalla, colores, y variables iniciales para los objetos del juego. Se importan modelos de IA (Red Neuronal, Árbol de Decisión, KNN) y se configuran variables clave

```
# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Phaser")

# Variables del jugador, balas, nave, fondo, etc.
jugador = None
bala = None
bala_vertical = None
fondo = None
nave = None

from redNeuronal import (...)
from decisionTree import (...)
from knn import (...)

modelos_cargados = {
    "nn": {"salto": False, "movimiento": False},
    "dt": {"salto": False, "movimiento": False},
    "knn": {"salto": False, "movimiento": False}
}

modelos_cargados["nn"]["salto"] = cargarRedSalto()
modelos_cargados["dt"]["salto"] = cargar_modelo_salto_dt()
...

```

#### Modelos de IA
Predicción de Acciones
La función aplicar_ia() utiliza el modelo seleccionado anteriormente en el menú para predecir la accion de saltos y movimientos:

```

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

```


#### Recolección de datos
Datos para el Modelo de Salto. Características:
+ velocidad_bala: Velocidad de la bala horizontal (para determinar urgencia).
+ distancia: Distancia horizontal entre el jugador y la bala (para evaluar riesgo de colisión).

target:
+ salto_hecho: 1 si el jugador saltó, 0 si no lo hizo.

Predecir cuándo saltar en función de la velocidad y proximidad de la bala horizontal.

Datos para el Modelo de Movimiento. Características:
+ jugador.x, jugador.y: Posición actual del jugador.
+ bala_vertical.x, bala_vertical.y: Posición de la bala vertical.
+ distancia_x, distancia_y: Distancia relativa entre el jugador y la bala vertical.
+ bala_vertical_disparada: 1 si hay una bala activa, 0 si no.

Etiqueta (target):
+ pos_actual: Acción tomada (0=izquierda, 1=quieto, 2=derecha).

Moverse lateralmente para esquivar balas verticales, considerando posiciones y distancias.

```

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

```

#### Entrenamiento de Modelos
Se modifico el menú original para poder entrenar los modelos de manera individual o al mismo tiempo todos. Con ello llamando a su respectivo archivo donde se encuentran los métodos específicos para el entrenmiento de cada modelo.


```

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

```

### Modelos 

#### Red Neuronal
Este modelo utiliza una arquitectura de red neuronal profunda, con siguientes caracteristicas por red.

Para salto:
+ Es una red binaria (sigmoid en la capa final) que predice la probabilidad de saltar.
+ Entrada: 2 características (velocidad_bala, distancia).
+ Capas ocultas: 3 capas densas (relu) con 32, 32 y 16 neuronas.
+ Entrenamiento: 70 épocas, optimizador adam, pérdida binary_crossentropy.

Para movimiento:
+ Es una red multiclase (softmax en la capa final) que predice dirección (izquierda, quieto, derecha).
+ Entrada: 7 características (posición del jugador, bala, distancias).
+ Capas ocultas: 2 capas densas (relu) con 64 y 32 neuronas.
+ Entrenamiento: 100 épocas, optimizador adam, pérdida categorical_crossentropy.

```
import random
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Variables globales para guardar los modelos
modeloSaltoEntrenado = None
modeloMovimientoEntrenado = None

# Archivos para guardar modelos
rutaBase = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2_v2(Phaser_Modelos)\\modelosGuardados'
archivoModeloSalto = os.path.join(rutaBase, 'nn_modelo_salto.h5')
archivoModeloMovimiento = os.path.join(rutaBase, 'nn_modelo_movimiento.h5')

def entrenarRed(datosEntrada):
    if len(datosEntrada) < 10:
        print("Insuficientes datos para entrenar el modelo.")
        return None

    informacion = np.array(datosEntrada)
    caracteristicas = informacion[:, :2]  # Velocidad y distancia
    etiquetas = informacion[:, 2]   # Salto
    caracteristicasEntrenamiento, caracteristicasPrueba, etiquetasEntrenamiento, etiquetasPrueba = train_test_split(
        caracteristicas, etiquetas, test_size=0.2, random_state=42)

    redNeural = Sequential([
        Dense(32, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    redNeural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    redNeural.fit(caracteristicasEntrenamiento, etiquetasEntrenamiento, epochs=70, batch_size=32, verbose=1)
    perdida, precision = redNeural.evaluate(caracteristicasPrueba, etiquetasPrueba, verbose=0)
    print(f"Modelo entrenado con precisión: {precision:.2f}")
    
    return redNeural

def determinarSalto(jugador, bala, velocidadBala, balaAire, balaDisparadaAire, modeloEntrenado, salto, enSuelo):
    if modeloEntrenado is None:
        print("Modelo no entrenado. No se puede decidir.")
        return False, enSuelo

    distanciaSuelo = abs(jugador.x - bala.x)
    distanciaAireX = abs(jugador.centerx - balaAire.centerx)
    distanciaAireY = abs(jugador.centery - balaAire.centery)
    existeBalaAire = 1 if balaDisparadaAire else 0

    datosEntrada = np.array([[velocidadBala, distanciaSuelo, distanciaAireX, distanciaAireY, existeBalaAire, jugador.x]])

    resultado = modeloEntrenado.predict(datosEntrada, verbose=0)[0][0]

    if resultado > 0.5 and enSuelo:
        salto = True
        enSuelo = False
        print("Saltar")

    return salto, enSuelo

def entrenarRedDesplazamiento(datosMovimiento):
    if len(datosMovimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    informacion = np.array(datosMovimiento)
    caracteristicas = informacion[:, :7].astype('float32')
    etiquetas = informacion[:, 7].astype('int')       # izquierda, quieto, derecha

    etiquetasCategoricas = to_categorical(etiquetas, num_classes=3)

    caracteristicasEntrenamiento, caracteristicasPrueba, etiquetasEntrenamiento, etiquetasPrueba = train_test_split(
        caracteristicas, etiquetasCategoricas, test_size=0.2, random_state=42)

    redNeural = Sequential([
        Dense(64, input_dim=7, activation='relu'), 
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    redNeural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    redNeural.fit(caracteristicasEntrenamiento, etiquetasEntrenamiento, epochs=100, batch_size=32, verbose=1)

    perdida, precision = redNeural.evaluate(caracteristicasPrueba, etiquetasPrueba, verbose=0)
    print(f"Precisión del modelo de movimiento: {precision:.2f}")
    
    return redNeural

def determinarMovimiento(jugador, bala, modeloMovimiento, salto, balaSuelo):
    if modeloMovimiento is None:
        print("Modelo no entrenado.")
        return jugador.x, 1  # Quieto por defecto

    distanciaBalaTerreno = abs(jugador.x - balaSuelo.x)
    # posicion x, altura y, bala aérea x, bala aérea y, bala del suelo x, bala del suelo y, distancia a bala del suelo, si está saltando
    datosEntrada = np.array([[
        jugador.x,            
        jugador.y,             
        bala.centerx,         
        bala.centery,          
        balaSuelo.x,            
        balaSuelo.y,             
        distanciaBalaTerreno,      
        1 if salto else 0            
    ]], dtype='float32')

    resultado = modeloMovimiento.predict(datosEntrada, verbose=0)[0]
    accionSeleccionada = np.argmax(resultado)

    if accionSeleccionada == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda")
    elif accionSeleccionada == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha")
    else:
        print("Quieto")

    return jugador.x, accionSeleccionada


# Variables para controlar el estado de los modelos
modeloSaltoCargado = False
modeloMovimientoCargado = False
modeloSalto = None
modeloMovimiento = None

def cargarRedSalto():
    global modeloSalto, modeloSaltoCargado
    
    try:
        if os.path.exists(archivoModeloSalto):
            modeloSalto = load_model(archivoModeloSalto)
            modeloSaltoCargado = True
            print(f"Modelo de salto cargado correctamente desde {archivoModeloSalto}")
            return True
        else:
            print("No se encontró el modelo de salto. Entrena primero en modo manual.")
            modeloSaltoCargado = False
            return False
    except Exception as e:
        print(f"Error al cargar el modelo de salto: {e}")
        modeloSaltoCargado = False
        return False

def entrenarRedSalto(datosSalto):
    global modeloSalto, modeloSaltoCargado
    
    if len(datosSalto) < 10:
        print("No hay suficientes datos para entrenar el modelo de salto.")
        return False
    
    print("Entrenando modelo de salto...")
    modeloSalto = entrenarRed(datosSalto)
    
    if modeloSalto is not None:
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(archivoModeloSalto), exist_ok=True)
        # Guardar el modelo entrenado
        modeloSalto.save(archivoModeloSalto)
        modeloSaltoCargado = True
        print(f"Modelo de salto entrenado y guardado en {archivoModeloSalto}")
        return True
    
    return False

def pronosticarSalto(velocidadBala, distancia):
    global modeloSalto
    
    if not modeloSaltoCargado or modeloSalto is None:
        return 0.0
    
    try:
        datosEntrada = np.array([[velocidadBala, distancia]])
        resultado = modeloSalto.predict(datosEntrada, verbose=0)[0][0]
        return resultado
    except Exception as e:
        print(f"Error al predecir salto: {e}")
        return 0.0

def cargarRedMovimiento():
    global modeloMovimiento, modeloMovimientoCargado
    
    try:
        if os.path.exists(archivoModeloMovimiento):
            modeloMovimiento = load_model(archivoModeloMovimiento)
            modeloMovimientoCargado = True
            print(f"Modelo de movimiento cargado correctamente desde {archivoModeloMovimiento}")
            return True
        else:
            print("No se encontró el modelo de movimiento. Entrena primero en modo manual.")
            modeloMovimientoCargado = False
            return False
    except Exception as e:
        print(f"Error al cargar el modelo de movimiento: {e}")
        modeloMovimientoCargado = False
        return False

def entrenarRedMovimiento(datosMovimiento):
    global modeloMovimiento, modeloMovimientoCargado
    
    if len(datosMovimiento) < 10:
        print("No hay suficientes datos para entrenar el modelo de movimiento.")
        return False
    
    print("Entrenando modelo de movimiento...")
    
    movimientos = [d[7] for d in datosMovimiento]
    izquierda = movimientos.count(0)
    estatico = movimientos.count(1)
    derecha = movimientos.count(2)
    print(f"Distribución original: Izq={izquierda}({izquierda/len(movimientos):.1%}), Quieto={estatico}({estatico/len(movimientos):.1%}), Der={derecha}({derecha/len(movimientos):.1%})")
    
    informacionEquilibrada = []
    for dato in datosMovimiento:
        if dato[7] != 1 or random.random() < 0.5:  # reduce sesgo de quieto
            informacionEquilibrada.append(dato)
    
    modeloMovimiento = entrenarRedDesplazamiento(informacionEquilibrada)
    
    if modeloMovimiento is not None:
        os.makedirs(os.path.dirname(archivoModeloMovimiento), exist_ok=True)
        modeloMovimiento.save(archivoModeloMovimiento)
        modeloMovimientoCargado = True
        print(f"Modelo de movimiento entrenado y guardado en {archivoModeloMovimiento}")
        return True
    
    return False

def pronosticarMovimiento(jugadorX, jugadorY, balaVerticalX, balaVerticalY, balaActiva):
    global modeloMovimiento
    
    if not modeloMovimientoCargado or modeloMovimiento is None:
        return 1 
    
    try:
        distanciaX = jugadorX - balaVerticalX
        distanciaY = jugadorY - balaVerticalY
        
        datosEntrada = np.array([[
            jugadorX, jugadorY,
            balaVerticalX, balaVerticalY,
            distanciaX, distanciaY,
            1 if balaActiva else 0
        ]], dtype='float32')
        
        resultado = modeloMovimiento.predict(datosEntrada, verbose=0)[0]
        print(f"Probabilidades: Izq={resultado[0]:.2f}, Quieto={resultado[1]:.2f}, Der={resultado[2]:.2f}")

            
        return np.argmax(resultado)
    except Exception as e:
        print(f"Error al predecir movimiento: {e}")
        return 1
```

#### Árbol de Decisión
Para salto y movimiento:
+ Entrada: Mismos datos que la red neuronal.
+ Configuración: max_depth=5 (limita profundidad para evitar sobreajuste), random_state=42.
+ Funcionamiento: Divide los datos en nodos según características (ej: distancia > 30 → saltar).

```
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_PATH = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2_v2(Phaser_Modelos)\\modelosGuardados'
MODELO_SALTO_PATH = os.path.join(BASE_PATH, 'dt_modelo_salto.pkl')
MODELO_MOVIMIENTO_PATH = os.path.join(BASE_PATH, 'dt_modelo_movimiento.pkl')
SCALER_SALTO_PATH = os.path.join(BASE_PATH, 'dt_scaler_salto.pkl')
SCALER_MOVIMIENTO_PATH = os.path.join(BASE_PATH, 'dt_scaler_movimiento.pkl')

# Variables globales para los modelos
dt_modelo_salto = None
dt_modelo_movimiento = None
dt_scaler_salto = None
dt_scaler_movimiento = None

def entrenar_modelo_salto_dt(datos_salto):
    global dt_modelo_salto, dt_scaler_salto
    
    if len(datos_salto) < 10:
        print("No hay suficientes datos para entrenar el modelo de salto (DT)")
        return False
    
    try:
        # Convertir a numpy array
        datos = np.array(datos_salto)
        X = datos[:, :2]  # velocidad de bala y distancia
        y = datos[:, 2]   # salto 
        
        # Escalar los datos
        dt_scaler_salto = StandardScaler()
        X_scaled = dt_scaler_salto.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Crear y entrenar el modelo
        dt_modelo_salto = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_modelo_salto.fit(X_train, y_train)
        
        # Evaluar el modelo
        score = dt_modelo_salto.score(X_test, y_test)
        print(f"Precisión del modelo de salto (DT): {score:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(MODELO_SALTO_PATH), exist_ok=True)
        joblib.dump(dt_modelo_salto, MODELO_SALTO_PATH)
        joblib.dump(dt_scaler_salto, SCALER_SALTO_PATH)
        
        return True
    
    except Exception as e:
        print(f"Error al entrenar el modelo de salto (DT): {e}")
        return False

def entrenar_modelo_movimiento_dt(datos_movimiento):
    global dt_modelo_movimiento, dt_scaler_movimiento
    
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el modelo de movimiento (DT)")
        return False
    
    try:
        # Convertir a numpy array
        datos = np.array(datos_movimiento)
        X = datos[:, :7]  # primeras 7 columnas son características
        y = datos[:, 7]   # última columna es la acción (0, 1, 2)
        
        # Escalar los datos
        dt_scaler_movimiento = StandardScaler()
        X_scaled = dt_scaler_movimiento.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Crear y entrenar el modelo
        dt_modelo_movimiento = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_modelo_movimiento.fit(X_train, y_train)
        
        # Evaluar el modelo
        score = dt_modelo_movimiento.score(X_test, y_test)
        print(f"Precisión del modelo de movimiento (DT): {score:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(MODELO_MOVIMIENTO_PATH), exist_ok=True)
        joblib.dump(dt_modelo_movimiento, MODELO_MOVIMIENTO_PATH)
        joblib.dump(dt_scaler_movimiento, SCALER_MOVIMIENTO_PATH)
        
        return True
    
    except Exception as e:
        print(f"Error al entrenar el modelo de movimiento (DT): {e}")
        return False

def cargar_modelo_salto_dt():
    global dt_modelo_salto, dt_scaler_salto
    
    try:
        if os.path.exists(MODELO_SALTO_PATH) and os.path.exists(SCALER_SALTO_PATH):
            dt_modelo_salto = joblib.load(MODELO_SALTO_PATH)
            dt_scaler_salto = joblib.load(SCALER_SALTO_PATH)
            print("Modelo de salto (DT) cargado correctamente")
            return True
        else:
            print("No se encontró el modelo de salto (DT) guardado")
            return False
    
    except Exception as e:
        print(f"Error al cargar el modelo de salto (DT): {e}")
        return False

def cargar_modelo_movimiento_dt():
    global dt_modelo_movimiento, dt_scaler_movimiento
    
    try:
        if os.path.exists(MODELO_MOVIMIENTO_PATH) and os.path.exists(SCALER_MOVIMIENTO_PATH):
            dt_modelo_movimiento = joblib.load(MODELO_MOVIMIENTO_PATH)
            dt_scaler_movimiento = joblib.load(SCALER_MOVIMIENTO_PATH)
            print("Modelo de movimiento (DT) cargado correctamente")
            return True
        else:
            print("No se encontró el modelo de movimiento (DT) guardado")
            return False
    
    except Exception as e:
        print(f"Error al cargar el modelo de movimiento (DT): {e}")
        return False

def predecir_salto_dt(velocidad_bala, distancia):
    global dt_modelo_salto, dt_scaler_salto
    
    if dt_modelo_salto is None or dt_scaler_salto is None:
        print("El modelo de salto (DT) no está cargado")
        return 0.0
    
    try:
        # Transformar la entrada
        entrada = np.array([[velocidad_bala, distancia]])
        entrada_scaled = dt_scaler_salto.transform(entrada)
        
        # Predecir probabilidad de clase positiva
        prob = dt_modelo_salto.predict_proba(entrada_scaled)[0][1]
        return prob
    
    except Exception as e:
        print(f"Error al predecir salto (DT): {e}")
        return 0.0

def predecir_movimiento_dt(jugador_x, jugador_y, bala_x, bala_y, bala_activa):
    global dt_modelo_movimiento, dt_scaler_movimiento
    
    if dt_modelo_movimiento is None or dt_scaler_movimiento is None:
        print("El modelo de movimiento (DT) no está cargado")
        return 1  # Por defecto, quedarse quieto
    
    try:
        # Calcular distancias
        distancia_x = jugador_x - bala_x
        distancia_y = jugador_y - bala_y
        
        # Transformar la entrada
        entrada = np.array([[
            jugador_x, jugador_y,
            bala_x, bala_y,
            distancia_x, distancia_y,
            1 if bala_activa else 0
        ]])
        entrada_scaled = dt_scaler_movimiento.transform(entrada)
        
        # Predecir acción
        accion = dt_modelo_movimiento.predict(entrada_scaled)[0]
        return int(accion)
    
    except Exception as e:
        print(f"Error al predecir movimiento (DT): {e}")
        return 1  # Por defecto, quedarse quieto
```

#### K-Nearest Neighbors (KNN)
Para salto y movimiento:
+ Entrada: Mismos datos que los anteriores.
+ Configuración: k dinámico (calculado como sqrt(n_datos)), métrica de distancia euclidiana.
+ Escalado: Usa StandardScaler para normalizar características.
+ Funcionamiento: Compara la instancia actual con ejemplos históricos para decidir la acción.

```
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_PATH = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2_v2(Phaser_Modelos)\\modelosGuardados'
MODELO_SALTO_PATH = os.path.join(BASE_PATH, 'knn_modelo_salto.pkl')
MODELO_MOVIMIENTO_PATH = os.path.join(BASE_PATH, 'knn_modelo_movimiento.pkl')
SCALER_SALTO_PATH = os.path.join(BASE_PATH, 'knn_scaler_salto.pkl')
SCALER_MOVIMIENTO_PATH = os.path.join(BASE_PATH, 'knn_scaler_movimiento.pkl')

# Variables globales para los modelos
knn_modelo_salto = None
knn_modelo_movimiento = None
knn_scaler_salto = None
knn_scaler_movimiento = None

def entrenar_modelo_salto_knn(datos_salto):
    global knn_modelo_salto, knn_scaler_salto
    
    if len(datos_salto) < 10:
        print("No hay suficientes datos para entrenar el modelo de salto (KNN)")
        return False
    
    try:
        # Convertir a numpy array
        datos = np.array(datos_salto)
        X = datos[:, :2]  # velocidad_bala y distancia
        y = datos[:, 2]   # salto (0 o 1)
        
        # Escalar los datos
        knn_scaler_salto = StandardScaler()
        X_scaled = knn_scaler_salto.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Determinar un buen k (número de vecinos)
        k = min(15, max(3, int(np.sqrt(len(datos_salto)))))
        print(f"Usando k={k} para el modelo de salto")
        
        # Crear y entrenar el modelo
        knn_modelo_salto = KNeighborsClassifier(n_neighbors=k)
        knn_modelo_salto.fit(X_train, y_train)
        
        # Evaluar el modelo
        score = knn_modelo_salto.score(X_test, y_test)
        print(f"Precisión del modelo de salto (KNN): {score:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(MODELO_SALTO_PATH), exist_ok=True)
        joblib.dump(knn_modelo_salto, MODELO_SALTO_PATH)
        joblib.dump(knn_scaler_salto, SCALER_SALTO_PATH)
        
        return True
    
    except Exception as e:
        print(f"Error al entrenar el modelo de salto (KNN): {e}")
        return False

def entrenar_modelo_movimiento_knn(datos_movimiento):
    global knn_modelo_movimiento, knn_scaler_movimiento
    
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el modelo de movimiento (KNN)")
        return False
    
    try:
        # Convertir a numpy array
        datos = np.array(datos_movimiento)
        X = datos[:, :7]  # primeras 7 columnas son características
        y = datos[:, 7]   # última columna es la acción (0, 1, 2)
        
        # Escalar los datos
        knn_scaler_movimiento = StandardScaler()
        X_scaled = knn_scaler_movimiento.fit_transform(X)
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Determinar un buen k (número de vecinos)
        k = min(15, max(3, int(np.sqrt(len(datos_movimiento)))))
        print(f"Usando k={k} para el modelo de movimiento")
        
        # Crear y entrenar el modelo
        knn_modelo_movimiento = KNeighborsClassifier(n_neighbors=k)
        knn_modelo_movimiento.fit(X_train, y_train)
        
        # Evaluar el modelo
        score = knn_modelo_movimiento.score(X_test, y_test)
        print(f"Precisión del modelo de movimiento (KNN): {score:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(MODELO_MOVIMIENTO_PATH), exist_ok=True)
        joblib.dump(knn_modelo_movimiento, MODELO_MOVIMIENTO_PATH)
        joblib.dump(knn_scaler_movimiento, SCALER_MOVIMIENTO_PATH)
        
        return True
    
    except Exception as e:
        print(f"Error al entrenar el modelo de movimiento (KNN): {e}")
        return False

def cargar_modelo_salto_knn():
    global knn_modelo_salto, knn_scaler_salto
    
    try:
        if os.path.exists(MODELO_SALTO_PATH) and os.path.exists(SCALER_SALTO_PATH):
            knn_modelo_salto = joblib.load(MODELO_SALTO_PATH)
            knn_scaler_salto = joblib.load(SCALER_SALTO_PATH)
            print("Modelo de salto (KNN) cargado correctamente")
            return True
        else:
            print("No se encontró el modelo de salto (KNN) guardado")
            return False
    
    except Exception as e:
        print(f"Error al cargar el modelo de salto (KNN): {e}")
        return False

def cargar_modelo_movimiento_knn():
    global knn_modelo_movimiento, knn_scaler_movimiento
    
    try:
        if os.path.exists(MODELO_MOVIMIENTO_PATH) and os.path.exists(SCALER_MOVIMIENTO_PATH):
            knn_modelo_movimiento = joblib.load(MODELO_MOVIMIENTO_PATH)
            knn_scaler_movimiento = joblib.load(SCALER_MOVIMIENTO_PATH)
            print("Modelo de movimiento (KNN) cargado correctamente")
            return True
        else:
            print("No se encontró el modelo de movimiento (KNN) guardado")
            return False
    
    except Exception as e:
        print(f"Error al cargar el modelo de movimiento (KNN): {e}")
        return False

def predecir_salto_knn(velocidad_bala, distancia):
    global knn_modelo_salto, knn_scaler_salto
    
    if knn_modelo_salto is None or knn_scaler_salto is None:
        print("El modelo de salto (KNN) no está cargado")
        return 0.0
    
    try:
        # Transformar la entrada
        entrada = np.array([[velocidad_bala, distancia]])
        entrada_scaled = knn_scaler_salto.transform(entrada)
        
        # Predecir probabilidad de clase positiva
        prob = knn_modelo_salto.predict_proba(entrada_scaled)[0][1]
        return prob
    
    except Exception as e:
        print(f"Error al predecir salto (KNN): {e}")
        return 0.0

def predecir_movimiento_knn(jugador_x, jugador_y, bala_x, bala_y, bala_activa):
    global knn_modelo_movimiento, knn_scaler_movimiento
    
    if knn_modelo_movimiento is None or knn_scaler_movimiento is None:
        print("El modelo de movimiento (KNN) no está cargado")
        return 1  # Por defecto, quedarse quieto
    
    try:
        # Calcular distancias
        distancia_x = jugador_x - bala_x
        distancia_y = jugador_y - bala_y
        
        # Transformar la entrada
        entrada = np.array([[
            jugador_x, jugador_y,
            bala_x, bala_y,
            distancia_x, distancia_y,
            1 if bala_activa else 0
        ]])
        entrada_scaled = knn_scaler_movimiento.transform(entrada)
        
        # Predecir acción
        accion = knn_modelo_movimiento.predict(entrada_scaled)[0]
        return int(accion)
    
    except Exception as e:
        print(f"Error al predecir movimiento (KNN): {e}")
        return 1  # Por defecto, quedarse quieto
```