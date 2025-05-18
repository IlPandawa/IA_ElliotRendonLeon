import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .modeloInterface import ModelInterface

class LogisticRegressionModel(ModelInterface):
    def __init__(self):
        super().__init__()
        self.model_name = "Regresión Logística"
        self.model = None
        self.scaler = None
        self.model_file = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2(pyGame Phaser)\\models\\modelosGuardados\\lr_model.pkl'
        self.scaler_file = 'C:\\InteligenciaArtificial\\IA_ElliotRendonLeon\\Proyecto2(pyGame Phaser)\\models\\modelosGuardados\\lr_scaler.pkl'

    def train(self, data):
        try:
            print(f"Entrenando {self.model_name}...")
            
            # Convertir datos a DataFrame
            df = pd.DataFrame(data, columns=['velocidad_bala', 'distancia', 'salto'])
            X = df[['velocidad_bala', 'distancia']]
            y = df['salto']
            
            # Escalar características
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Crear y entrenar modelo
            self.model = LogisticRegression(random_state=42)
            self.model.fit(X_scaled, y)
            
            # Guardar modelo y scaler
            self.save()
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error al entrenar {self.model_name}: {e}")
            return False
    
    def predict(self, velocidad_bala, distancia):
        if not self.is_trained or self.model is None or self.scaler is None:
            print(f"{self.model_name} no entrenado o no cargado")
            return 0.0
        
        try:
            entrada = np.array([[velocidad_bala, distancia]])
            entrada_esc = self.scaler.transform(entrada)
            # Obtener probabilidad de clase positiva
            return float(self.model.predict_proba(entrada_esc)[0][1])  # Probabilidad de clase 1 (salto)
        except Exception as e:
            print(f"Error al predecir con {self.model_name}: {e}")
            return 0.0
    
    def load(self):
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = joblib.load(self.model_file)
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
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print(f"{self.model_name} guardado correctamente")
            return True
        except Exception as e:
            print(f"Error al guardar {self.model_name}: {e}")
            return False