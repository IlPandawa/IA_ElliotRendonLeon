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