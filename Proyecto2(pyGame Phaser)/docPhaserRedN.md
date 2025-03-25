# Documentación: Implementación de modo automático en Phaser

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

2. `entrenarModelo()`: Esta parte tiene como objetivo entrenar la red neuronal con los datos recolectados. Se crea un DataFrame con los datos de velocidad_bala, distancia y salto, se escalan las características y se normalizan los datos transformando los valores en el rango [0,1] *(Esto para evitar errores de si por ejemplo la velocidad es -7 y la distancia 300, se normalizan los valores transformandolos entre 0-1)*. Se define la red con 2 capas, la capa oculta tiene 8 neuronas y se activa mediante la función de activación ReLU, y en la capa de salida tiene una neurona con la función de activación sigmoide para arrojar valores en la probabilidad de salto entre 0-1. Se compila y se entrena durante 30 épocas, por ultimo se guarda el modelo utilizando joblib.
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

3. `predecirSalto()`: Esta función calcula la probabilidad de realizar un salto dependiendo de distancia entre el jugador y la bala utilizando la probabilidad del modelo. Se crea un DataFrame con la información de la velocidad y distancia actuales de la bala, se normalizan los datos con el escalador y se realiza la predicción para que nos arroje una probabilidad entre 0-1.
```
#* función para predecir salto
def predecirSalto():
    global jugador, bala, velocidad_bala
    distancia = abs(jugador.x - bala.x)
    
    entrada = pd.DataFrame([[velocidad_bala, distancia]], 
                         columns=['velocidad_bala', 'distancia'])
    try: 
        entrada_esc = escalador.transform(entrada)
        return modeloRedNeuronal.predict(entrada_esc, verbose=0)[0][0]
    except Exception as e:
        print("Error al predecir:", e)
        return 0.0
```

4. `modoAutoRed()`: Esta función es el pilar de tomar la desición entre hacer el salto o no, obteniendo la probabilidad y pregunta si el umbral es mayor a 0.49 decide si activar o no el salto dependiendo d ela predicción hecha por el modelo. *Se hicieron unos ajustes para mejorar el rendimiento que corresponden a las lineas de tiempo actual y ultima predicción, pero lo retomaremos más adelante explicado un poco más a detalle, puesto que fue un añadido posterior*
```
#* modificar el modo auto
def modoAutoRed():
    global salto, en_suelo, ULTIMA_PREDICCION
    tiempoActual = pygame.time.get_ticks() / 1000
    if (tiempoActual - ULTIMA_PREDICCION) > INTERVALO_PREDICCION:
        if not modeloEntrenado or not escalador:
            print("Modelo no entrenado, juega en modo manual")
            return
        
        if en_suelo:
            prob = predecirSalto()
            print(f"Probabilidad de salto: {prob}", end='\r')
            if prob > 0.49:  # Umbral más bajo
                salto = True
                en_suelo = False
        ULTIMA_PREDICCION = tiempoActual
        

```

---
## Integración en el juego
Para la integración del modelo en el juego base, s emodificaron algunos bloques existentes del código, primeramente en `mostrar_menu()` se agrego una tecla para entrenar el modelo y una validación para que el jugador no pueda entrar al modo auto sin antes haber jugado. Además de una modificación en el bucle principal `main()` para llamar a cargar el modelo y agregar la parte del modo auto.
```
# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO)
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, 'T' para entrenar el modelo, o 'Q' para Salir", True, BLANCO)
    pantalla.blit(texto, (w // 6, h // 3))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    if modeloEntrenado:
                        modo_auto = True
                        menu_activo = False
                    else :
                        print("Entrena primero el modelo jugando en manual o presiona 'T' ")
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_t:
                    entrenarModelo()
                    
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

def main():
    global salto, en_suelo, bala_disparada
    cargarModeloEntrenado()

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()
            #* modo auto
            if modo_auto:
                if modeloEntrenado:
                    modoAutoRed()
                    if salto:
                        manejar_salto()
            
            if not pausa:
                if modo_auto:
                    if modeloEntrenado:
                        modoAutoRed()
                    else:
                        print("Entrena primero el modelo jugando en manual")
            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()                    
```

## Optimización de rendimiento
Al iniciar el juego en modo automatico el rendimiento del juego empeora, esto sucede por las predicciones que tiene que realizar el modelo en tiempo de ejecución, la solución que encontré para mejorar un poco esta parte fue el de regular la frecuencia con la que se realizaban las predicciones por segundo, haciendo que se hagan las predicciones mínimo cada 0.2s. Esto se implemento en `modoAutoRed()`. Primero obteniendo el tiempo actual en segundos, posteriormente haciendo la pregunta si ya pasaron por lo menos 200 milisegundos antes de realizar la última predicción y posteriormente reiniciando el contador. Mejorando un poco el rendimiento sin que se vea compremetida la funcionalidad del modelo

```
#! mejora rendimiento
ULTIMA_PREDICCION = 0  # Tiempo de la ultima predicción
INTERVALO_PREDICCION = 0.2  # Segundos entre predicciones

#* modificar el modo auto
def modoAutoRed():
    global salto, en_suelo, ULTIMA_PREDICCION
    tiempoActual = pygame.time.get_ticks() / 1000
    if (tiempoActual - ULTIMA_PREDICCION) > INTERVALO_PREDICCION:
        if not modeloEntrenado or not escalador:
            print("Modelo no entrenado, juega en modo manual")
            return
        
        if en_suelo:
            prob = predecirSalto()
            print(f"Probabilidad de salto: {prob}", end='\r')
            if prob > 0.49:  # Umbral más bajo
                salto = True
                en_suelo = False
        ULTIMA_PREDICCION = tiempoActual
```

