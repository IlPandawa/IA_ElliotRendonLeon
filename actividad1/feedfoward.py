from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Cargar el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target


# Dividir en conjunto de entrenamiento y prueba (80% y 20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características (importante para MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir la red neuronal con una capa oculta de 10 neuronas
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500, random_state=42) #Preparacion del modelo

# Entrenar el modelo
mlp.fit(X_train, Y_train)

# Hacer predicciones
Y_pred = mlp.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(Y_test, Y_pred)
print(f'\nPrecisión en test: {accuracy: 4f}')
print(Y_pred)

#datos con los que se probaron
print(X_test)

# Guardar el modelo y el scaler para su uso posterior
joblib.dump(mlp, 'mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModelo y scaler guardados correctamente.")