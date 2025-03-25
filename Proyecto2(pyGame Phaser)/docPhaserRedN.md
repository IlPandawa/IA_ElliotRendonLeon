# Documentación: Implementación de modo automático

Se implemento el modo automático en el juego phaser utilizando un sistema de aprendizaje que permite al juego aprender de el usuario. Este aprendizaje lo realiza a través de una red neuronal que después de haber sido entrenada toma la desición de saltar y esquivar la bala basado en los datos recolectados in-game.

--- 
## Librerias utilizadas
Adicional a pyGame para los gráficos y la gestión de eventos, se utilizaron las siguientes librerias:

- **Tensorflow/Keras**: Libreria para la inicialización y entrenamiento de la red neuronal
- **Scikit-Learn**: Librería para el procesamiento de datos
- **Joblib**: Librería para realizar el guardado del modelo para su posterior reutilización.
- **Pandas**: Librería para la manipulación de los datos de entrenamiento

--- 
## Explicación del código
Variables para gestionar el funcionamiento del modelo
```
modeloRedNeuronal = None       # almacena el modelo entrenado
escalador = None               # para normalización de datos
modeloEntrenado = False        # bandera para verificar si hay modelo cargado
ARCHIVO_MODELO = 'modeloJuego.pkl'     # Nombre del modelo guardado
ARCHIVO_ESCALADOR = 'escaladorJuego.pkl' # Nombre del escalador
```

#### Funcionamiento del modelo
En estos bloques de código se carga si es que hay un modelo ya entrenado, o si no es el caso lo entrena. Estas son las funciones clave:

1. `cargarModeloEntrenado()`: Este bloque se encarga de cargar los archivos .pkl para su reutilización si el modelo ya esta entrenado, funciona primeramente intentando cargar los archivos utilizando joblib, si es que existe se cambia la bandera a `modeloEntrenado = True`. En caso de no encontrar los archivos muestra un mensaje para mantener al jugador en modo manual
```
def cargarModeloEntrenado():
    global modeloRedNeuronal, escalador, modeloEntrenado

    try:
        modeloRedNeuronal = joblib.load(ARCHIVO_MODELO)
        escalador = joblib.load(ARCHIVO_ESCALADOR)
        modeloEntrenado=True
        print("Modelo cargado")
    except FileNotFoundError:
        print("No se encontro el modelo, entrena en modo manual")
        modeloEntrenado=False
```

2. `entrenarModelo()`: Esta parte tiene como objetivo entrenar la red neuronal con los datos recolectados. Se crea un DataFrame con los datos de velocidad_bala, distancia y salto, se escalan las características y se normalizan los datos transformando los valores en el rango [0,1]. Se define la red con 2 capas, la capa oculta tiene 8 neuronas y se activa mediante la función de activación ReLU, y en la capa de salida tiene una neurona con la función de activación sigmoide para arrojar valores en la probabilidad de salto entre 0-1. Se compila y se entrena durante 30 épocas, por ultimo se guarda el modelo utilizando joblib.
```
#* Entrenar el modelo y guardar en joblib
def entrenarModelo():
    global modeloRedNeuronal, escalador, modeloEntrenado

    print("Entrenando modelo...")

    #* procesamiento
    df = pd.DataFrame(datos_modelo, columns=['velocidad_bala', 'distancia', 'salto'])
    X = df[['velocidad_bala', 'distancia']] 
    y = df['salto']
    
    #* escalar las características
    escalador = MinMaxScaler()
    X_escalado = escalador.fit_transform(X)

    # Entrenamiento de modelo
    modeloRedNeuronal = Sequential([
        Dense(8, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ]) 
    #* compilar el modelo
    modeloRedNeuronal.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )

    #* entrenamiento
    modeloRedNeuronal.fit(
        X_escalado, y,
        epochs = 30,
        batch_size = 8,
        verbose = 1
    )

    #* guardar el modelo
    joblib.dump(modeloRedNeuronal, ARCHIVO_MODELO)
    joblib.dump(escalador, ARCHIVO_ESCALADOR)
    modeloEntrenado = True
    print("Modelo entrenado y guardado")
    
    return True
```

