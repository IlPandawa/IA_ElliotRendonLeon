import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import joblib
import os

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Variables globales para guardar los modelos
modelo_salto_entrenado = None
modelo_movimiento_entrenado = None

# Archivos para guardar modelos
ARCHIVO_MODELO_SALTO = 'modelo_salto.h5'
ARCHIVO_MODELO_MOVIMIENTO = 'modelo_movimiento.h5'


# ---------------- MODELO DE RED NEURONAL ----------------
# ---------------- RED NEURONAL DE SALTO -----------------
def entrenar_modelo(datos_modelo):
    if len(datos_modelo) < 10:
        print("Insuficientes datos para entrenar el modelo.")
        return None

    datos = np.array(datos_modelo)
    X = datos[:, :2]  # Velocidad y distancia
    y = datos[:, 2]   # Salto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = Sequential([
        Dense(32, input_dim=2, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo.fit(X_train, y_train, epochs=70, batch_size=32, verbose=1)
    loss, accuracy = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"Modelo entrenado con precisión: {accuracy:.2f}")
    
    return modelo

def decidir_salto(jugador, bala, velocidad_bala, bala_aire, bala_disparada_aire, modelo_entrenado, salto, en_suelo):
    if modelo_entrenado is None:
        print("Modelo no entrenado. No se puede decidir.")
        return False, en_suelo

    distancia_suelo = abs(jugador.x - bala.x)
    distancia_aire_x = abs(jugador.centerx - bala_aire.centerx)
    distancia_aire_y = abs(jugador.centery - bala_aire.centery)
    hay_bala_aire = 1 if bala_disparada_aire else 0

    entrada = np.array([[velocidad_bala, distancia_suelo, distancia_aire_x, distancia_aire_y, hay_bala_aire, jugador.x]])

    prediccion = modelo_entrenado.predict(entrada, verbose=0)[0][0]

    if prediccion > 0.5 and en_suelo:
        salto = True
        en_suelo = False
        print("Saltar")

    return salto, en_suelo


 
# ---------------- RED NEURONAL DE MOVIMIENTO -----------------
def entrenar_red_movimiento(datos_movimiento):
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar.")
        return None

    datos = np.array(datos_movimiento)
    X = datos[:, :7].astype('float32')
    y = datos[:, 7].astype('int')       # izquierda, quieto, derecha

    y_categorical = to_categorical(y, num_classes=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, input_dim=7, activation='relu'), 
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión del modelo de movimiento: {accuracy:.2f}")
    
    return model

def decidir_movimiento(jugador, bala, modelo_movimiento, salto, bala_suelo):
    if modelo_movimiento is None:
        print("Modelo no entrenado.")
        return jugador.x, 1  # Quieto por defecto

    distancia_bala_suelo = abs(jugador.x - bala_suelo.x)

    entrada = np.array([[
        jugador.x,                     # Posición actual del jugador
        jugador.y,                     # Altura del jugador
        bala.centerx,                  # X de la bala aérea
        bala.centery,                  # Y de la bala aérea
        bala_suelo.x,                  # X de la bala del suelo
        bala_suelo.y,                  # Y de la bala del suelo
        distancia_bala_suelo,          # Distancia a bala del suelo
        1 if salto else 0              # Si el jugador está saltando
    ]], dtype='float32')

    prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
    accion = np.argmax(prediccion)

    if accion == 0 and jugador.x > 0:
        jugador.x -= 5
        print("Izquierda")
    elif accion == 2 and jugador.x < 200 - jugador.width:
        jugador.x += 5
        print("Derecha")
    else:
        print("Quieto")

    return jugador.x, accion

# ---------------- FUNCIONES ADAPTADORAS PARA GAME.PY ----------------

# Variables para controlar el estado de los modelos
modelo_salto_cargado = False
modelo_movimiento_cargado = False
modelo_salto = None
modelo_movimiento = None

def cargar_modelo_salto():
    """Carga el modelo de salto desde archivo"""
    global modelo_salto, modelo_salto_cargado
    
    try:
        modelo_salto = load_model(ARCHIVO_MODELO_SALTO)
        modelo_salto_cargado = True
        print("Modelo de salto cargado correctamente")
        return True
    except (FileNotFoundError, OSError):
        print("No se encontró el modelo de salto. Entrena primero en modo manual.")
        modelo_salto_cargado = False
        return False

def cargar_modelo_movimiento():
    """Carga el modelo de movimiento desde archivo"""
    global modelo_movimiento, modelo_movimiento_cargado
    
    try:
        modelo_movimiento = load_model(ARCHIVO_MODELO_MOVIMIENTO)
        modelo_movimiento_cargado = True
        print("Modelo de movimiento cargado correctamente")
        return True
    except (FileNotFoundError, OSError):
        print("No se encontró el modelo de movimiento. Entrena primero en modo manual.")
        modelo_movimiento_cargado = False
        return False

def entrenar_modelo_salto(datos_salto):
    """Entrena el modelo para predecir cuándo saltar"""
    global modelo_salto, modelo_salto_cargado
    
    if len(datos_salto) < 10:
        print("No hay suficientes datos para entrenar el modelo de salto.")
        return False
    
    print("Entrenando modelo de salto...")
    modelo_salto = entrenar_modelo(datos_salto)
    
    if modelo_salto is not None:
        # Guardar el modelo entrenado
        modelo_salto.save(ARCHIVO_MODELO_SALTO)
        modelo_salto_cargado = True
        print("Modelo de salto entrenado y guardado")
        return True
    
    return False

def entrenar_modelo_movimiento(datos_movimiento):
    """Entrena el modelo para predecir a dónde moverse"""
    global modelo_movimiento, modelo_movimiento_cargado
    
    if len(datos_movimiento) < 10:
        print("No hay suficientes datos para entrenar el modelo de movimiento.")
        return False
    
    print("Entrenando modelo de movimiento...")
    
    # Balance de datos para evitar sesgo hacia "quedarse quieto"
    acciones = [d[7] for d in datos_movimiento]
    izq = acciones.count(0)
    quieto = acciones.count(1)
    der = acciones.count(2)
    print(f"Distribución original: Izq={izq}({izq/len(acciones):.1%}), Quieto={quieto}({quieto/len(acciones):.1%}), Der={der}({der/len(acciones):.1%})")
    
    # Seleccionar datos equilibrados (opcional)
    datos_balanceados = []
    for dato in datos_movimiento:
        # Incluir todos los movimientos y solo una parte de "quedarse quieto"
        if dato[7] != 1 or random.random() < 0.5:  # reduce "quieto" al 50%
            datos_balanceados.append(dato)
    
    modelo_movimiento = entrenar_red_movimiento(datos_balanceados)
    
    if modelo_movimiento is not None:
        # Guardar el modelo entrenado
        modelo_movimiento.save(ARCHIVO_MODELO_MOVIMIENTO)
        modelo_movimiento_cargado = True
        print("Modelo de movimiento entrenado y guardado")
        return True
    
    return False

def predecir_salto(velocidad_bala, distancia):
    """Predice si el jugador debe saltar basado en la velocidad y distancia"""
    global modelo_salto
    
    if not modelo_salto_cargado or modelo_salto is None:
        return 0.0
    
    try:
        entrada = np.array([[velocidad_bala, distancia]])
        prediccion = modelo_salto.predict(entrada, verbose=0)[0][0]
        return prediccion
    except Exception as e:
        print(f"Error al predecir salto: {e}")
        return 0.0

def predecir_movimiento(jugador_x, jugador_y, bala_vertical_x, bala_vertical_y, bala_activa):
    """Predice hacia dónde debe moverse el jugador"""
    global modelo_movimiento
    
    if not modelo_movimiento_cargado or modelo_movimiento is None:
        return 1  # Acción por defecto: quieto
    
    try:
        # Calcular distancias
        distancia_x = jugador_x - bala_vertical_x
        distancia_y = jugador_y - bala_vertical_y
        
        # Para el formato de entrada esperado por el modelo
        entrada = np.array([[
            jugador_x, jugador_y,
            bala_vertical_x, bala_vertical_y,
            distancia_x, distancia_y,
            1 if bala_activa else 0
        ]], dtype='float32')
        
        prediccion = modelo_movimiento.predict(entrada, verbose=0)[0]
        print(f"Probabilidades: Izq={prediccion[0]:.2f}, Quieto={prediccion[1]:.2f}, Der={prediccion[2]:.2f}")
        
        # Si las probabilidades son muy cercanas para movimiento vs quieto, favorecer el movimiento
        if max(prediccion[0], prediccion[2]) > prediccion[1] * 0.7:  # 70% del valor de "quieto"
            if prediccion[0] > prediccion[2]:
                return 0  # Izquierda
            else:
                return 2  # Derecha
        
        # Umbral elevado para la exploración para evitar quedarse atascado en "quieto"
        if random.random() < 0.15:  # 15% de exploración
            # Favorecer movimiento sobre quedarse quieto en la exploración
            return random.choice([0, 2])  # Solo izquierda o derecha para explorar
            
        return np.argmax(prediccion)
    except Exception as e:
        print(f"Error al predecir movimiento: {e}")
        return 1  # Por defecto: quieto