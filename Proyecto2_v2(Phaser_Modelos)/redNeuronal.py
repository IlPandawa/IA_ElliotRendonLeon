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