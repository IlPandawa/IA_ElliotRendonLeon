import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

from .modeloInterface import ModelInterface

class NeuralNetworkModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "Red Neuronal"
        self.model = None
        self.scaler = None
        self.model_file = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2(pyGame Phaser)\\models\\modelosGuardados\\nn_model.h5'
        self.scaler_file = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2(pyGame Phaser)\\models\\modelosGuardados\\nn_scaler.pkl'

    def train(self, data):
        """Entrenar el modelo de red neuronal"""
        try:
            print(f"Entrenando {self.model_name}...")
            
            # Convertir datos a DataFrame
            df = pd.DataFrame(data, columns=['velocidad_bala', 'distancia', 'salto'])
            X = df[['velocidad_bala', 'distancia']]
            y = df['salto']
            
            # Escalar características
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear y entrenar modelo
            self.model = Sequential([
                Dense(8, activation='relu', input_shape=(2,)),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model.fit(
                X_scaled, y,
                epochs=30,
                batch_size=8,
                verbose=1
            )
            
            # Guardar modelo y scaler
            self.save()
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error al entrenar {self.model_name}: {e}")
            return False
    
    def predict(self, velocidad_bala, distancia):
        """Hacer predicción con red neuronal"""
        if not self.is_trained or self.model is None or self.scaler is None:
            print(f"{self.model_name} no entrenado o no cargado")
            return 0.0
        
        try:
            entrada = pd.DataFrame([[velocidad_bala, distancia]], 
                                  columns=['velocidad_bala', 'distancia'])
            entrada_esc = self.scaler.transform(entrada)
            return float(self.model.predict(entrada_esc, verbose=0)[0][0])
        except Exception as e:
            print(f"Error al predecir con {self.model_name}: {e}")
            return 0.0
    
    def load(self):
        """Cargar modelo pre-entrenado de red neuronal"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = load_model(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                self.is_trained = True
                print(f"{self.model_name} cargado exitosamente")
                return True
            else:
                print(f"No se encontró modelo guardado para {self.model_name}")
                self.is_trained = False
                return False
        except Exception as e:
            print(f"Error al cargar {self.model_name}: {e}")
            self.is_trained = False
            return False
    
    def save(self):
        """Guardar modelo entrenado de red neuronal"""
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            self.model.save(self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print(f"{self.model_name} guardado correctamente")
            return True
        except Exception as e:
            print(f"Error al guardar {self.model_name}: {e}")
            return False