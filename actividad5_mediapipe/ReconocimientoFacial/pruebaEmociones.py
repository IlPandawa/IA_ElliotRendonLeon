import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Captura de video
cap = cv2.VideoCapture(0)

# Puntos faciales clave para detección
# Puntos para los párpados
puntosParpadoSup = [159, 145, 386, 374]  # Párpados superiores
puntosParpadoInf = [158, 153, 385, 373]  # Párpados inferiores

# Ojos, cejas, nariz y boca
puntosOjos = [33, 133, 362, 263, 159, 145, 386, 374]  # Contorno de ojos
puntosCejas = [65, 66, 70, 105, 107, 336, 296, 334]   # Puntos de cejas
puntosBoca = [61, 291, 0, 17, 13, 14, 78, 308]       # Contorno de boca
puntosNariz = [1, 2, 98, 327]                        # Puente y punta de la nariz

# Todos los puntos seleccionados
puntosFaciales = puntosOjos + puntosCejas + puntosBoca + puntosNariz + puntosParpadoSup + puntosParpadoInf

# Variables para detección de vida
historialMovimiento = deque(maxlen=30)  # Almacena movimiento reciente de puntos clave
ultimoTiempo = time.time()
ultimosPuntos = {}
umbralMovimiento = 0.3  # Umbral para considerar un movimiento significativo
contadorParpadeos = 0
ultimoEstadoOjos = "abiertos"
tiempoUltimoParpadeo = time.time()

# Variables para el parpadeo
parpadeoDetectado = False
contadorFramesCerrados = 0
umbralFramesCerrados = 2  # Número de frames consecutivos con ojos cerrados para contar como parpadeo

# Variables para almacenar las emociones recientes
historialEmociones = deque(maxlen=10)  # Almacena las últimas emociones detectadas
proporcionesPrevias = None

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detectarParpadeo(puntos, frame):
    """Detecta si los ojos están parpadeando"""
    global ultimoEstadoOjos, contadorParpadeos, tiempoUltimoParpadeo, parpadeoDetectado, contadorFramesCerrados

    puntosNecesarios = puntosParpadoSup + puntosParpadoInf
    if not all(p in puntos for p in puntosNecesarios):
        return False
    
    # Calcular distancia vertical entre párpados superiores e inferiores
    distanciaIzq = distancia(puntos[159], puntos[158]) + distancia(puntos[145], puntos[153])
    distanciaDer = distancia(puntos[386], puntos[385]) + distancia(puntos[374], puntos[373])
    
    # Calcular distancia horizontal de los ojos como referencia
    anchoOjoIzq = distancia(puntos[33], puntos[133])
    anchoOjoDer = distancia(puntos[362], puntos[263])
    
    # Calcular la relación de aspecto de los ojos (altura/anchura)
    relacionIzq = (distanciaIzq / 2) / anchoOjoIzq
    relacionDer = (distanciaDer / 2) / anchoOjoDer
    
    # Promedio de la relación de aspecto
    relacionPromedio = (relacionIzq + relacionDer) / 2
    
    umbralOjosCerrados = 0.11  #! ajustar despues
    
    # Determinar si los ojos están cerrados
    ojosCerrados = relacionPromedio < umbralOjosCerrados
    
    # Mostrar la relación de aspecto para depuración
    cv2.putText(frame, f"Relacion Ojos: {relacionPromedio:.3f}", (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if ojosCerrados:
        contadorFramesCerrados += 1
        estadoOjos = "cerrados"
        cv2.putText(frame, "Ojos: CERRADOS", (10, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    else:
        estadoOjos = "abiertos"
        cv2.putText(frame, "Ojos: ABIERTOS", (10, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Si los ojos se abrieron después de estar cerrados por suficientes frames
        if contadorFramesCerrados >= umbralFramesCerrados:
            contadorParpadeos += 1
            tiempoUltimoParpadeo = time.time()
            cv2.putText(frame, "Parpadeo Detectado!", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Reiniciar el contador de frames con ojos cerrados
        contadorFramesCerrados = 0
    
    ultimoEstadoOjos = estadoOjos
    return ojosCerrados

def calcularMovimiento(puntosPrevios, puntosActuales):
    """Calcula la cantidad de movimiento entre frames."""
    if not puntosPrevios:
        return 0
    
    movimientoTotal = 0
    contadorPuntos = 0
    
    for idx in puntosActuales:
        if idx in puntosPrevios:
            distanciaPuntos = distancia(puntosPrevios[idx], puntosActuales[idx])
            movimientoTotal += distanciaPuntos
            contadorPuntos += 1
    
    return movimientoTotal / max(contadorPuntos, 1)  # Evitar división por cero

def detectarPersonaViva(puntos, frame):
    """Determina si la persona está viva basándose en movimiento y parpadeos."""
    global ultimosPuntos, historialMovimiento, ultimoTiempo
    
    tiempoActual = time.time()
    tiempoPasado = tiempoActual - ultimoTiempo
    ultimoTiempo = tiempoActual
    
    # Calcular movimiento desde el último frame
    movimiento = calcularMovimiento(ultimosPuntos, puntos)
    historialMovimiento.append(movimiento)
    
    # Actualizar puntos para el siguiente frame
    ultimosPuntos = puntos.copy()
    
    # Calcular promedio de movimiento reciente
    movimientoPromedio = sum(historialMovimiento) / len(historialMovimiento) if historialMovimiento else 0
    
    # Verificar si hay parpadeo
    estaParpadeando = detectarParpadeo(puntos, frame)
    tiempoDesdeUltimoParpadeo = tiempoActual - tiempoUltimoParpadeo
    
    # Criterios para determinar si es una persona viva:
    # 1. Hay movimiento significativo en la cara
    # 2. Se detectan parpadeos periódicos
    esPersonaViva = (movimientoPromedio > umbralMovimiento or 
                     contadorParpadeos > 0 or 
                     tiempoDesdeUltimoParpadeo < 5.0)  # 5 segundos desde el último parpadeo
    
    # Mostrar información de prueba de vida
    cv2.putText(frame, f"Movimiento: {movimientoPromedio:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Parpadeos: {contadorParpadeos}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    estadoVida = "VIVA" if esPersonaViva else "POSIBLE FOTOGRAFÍA"
    colorEstado = (0, 255, 0) if esPersonaViva else (0, 0, 255)
    cv2.putText(frame, f"Persona: {estadoVida}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorEstado, 2)
    
    return esPersonaViva

def detectarEmocion(puntos, frame):
    """Detecta la emoción basándose en las proporciones faciales - algoritmo mejorado."""
    global proporcionesPrevias, historialEmociones
    
    # Verificar que todos los puntos necesarios estén disponibles
    puntosClave = [0, 17, 13, 14, 61, 291, 33, 133, 362, 263, 65, 66, 70, 105, 107, 336, 296, 334]
    if not all(p in puntos for p in puntosClave):
        return "Desconocida"
    
    # CÁLCULO DE PROPORCIONES FACIALES
    
    # 1. Apertura de la boca (distancia vertical entre labios)
    aperturaBoca = distancia(puntos[13], puntos[14])
    
    # 2. Anchura de la boca
    anchoBoca = distancia(puntos[61], puntos[291])
    
    # 3. Curvatura de la boca (comisuras respecto al centro)
    centroBocaY = (puntos[13][1] + puntos[14][1]) / 2
    comisuraIzqY = puntos[61][1]
    comisuraDerY = puntos[291][1]
    curvatura = ((centroBocaY - comisuraIzqY) + (centroBocaY - comisuraDerY)) / 2
    
    # 4. Posición de las cejas respecto a los ojos
    alturaCejaIzq = puntos[65][1] - puntos[159][1]  # Negativo si ceja elevada
    alturaCejaDer = puntos[296][1] - puntos[386][1]
    posicionCejas = (alturaCejaIzq + alturaCejaDer) / 2
    
    # 5. Distancia entre cejas (ceño fruncido)
    distanciaCejas = distancia(puntos[65], puntos[296])
    
    # 6. Apertura de los ojos
    aperturaOjoIzq = distancia(puntos[159], puntos[145])
    aperturaOjoDer = distancia(puntos[386], puntos[374])
    aperturaOjosPromedio = (aperturaOjoIzq + aperturaOjoDer) / 2
    
    # Normalización con distancia entre ojos como referencia
    distanciaEntreOjos = distancia(puntos[33], puntos[362])
    
    # Normalizar medidas
    aperturaBocaNorm = aperturaBoca / distanciaEntreOjos
    anchoBocaNorm = anchoBoca / distanciaEntreOjos
    curvaturaRelativa = curvatura / distanciaEntreOjos
    cejasPosicionRelativa = posicionCejas / distanciaEntreOjos
    distanciaCejasNorm = distanciaCejas / distanciaEntreOjos
    
    # Mostrar valores para depuración
    cv2.putText(frame, f"A.Boca: {aperturaBocaNorm:.2f} A.Ojos: {aperturaOjosPromedio:.2f}", (10, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Curvatura: {curvaturaRelativa:.2f} Cejas: {cejasPosicionRelativa:.2f}", (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # ALGORITMO MEJORADO DE DETECCIÓN DE EMOCIONES
    
    # Valores iniciales
    emocion = "Neutral"
    puntuacionFeliz = 0
    puntuacionTriste = 0
    puntuacionEnojado = 0
    puntuacionNeutral = 10  # Base para neutral
    
    # Características de emoción feliz:
    # - Comisuras elevadas (curvatura positiva)
    # - Boca más ancha
    if curvaturaRelativa > 0.02:  # Comisuras elevadas
        puntuacionFeliz += 15
    if anchoBocaNorm > 0.45:  # Boca ancha (sonrisa)
        puntuacionFeliz += 10
    
    # Características de emoción triste:
    # - Comisuras hacia abajo
    # - Cejas elevadas en los extremos
    if curvaturaRelativa < -0.01:  # Comisuras hacia abajo
        puntuacionTriste += 15
    if cejasPosicionRelativa < -0.02:  # Cejas elevadas
        puntuacionTriste += 10
    
    # Características de emoción enojada:
    # - Cejas juntas (ceño fruncido)
    # - Cejas bajas
    # - Boca tensa (menos ancha)
    if distanciaCejasNorm < 0.25:  # Cejas juntas
        puntuacionEnojado += 10
    if cejasPosicionRelativa > 0.02:  # Cejas bajas
        puntuacionEnojado += 15
    if anchoBocaNorm < 0.4 and aperturaBocaNorm < 0.1:  # Boca tensa
        puntuacionEnojado += 5
    
    # Determinar la emoción con mayor puntuación
    puntuaciones = {
        "Feliz": puntuacionFeliz,
        "Triste": puntuacionTriste,
        "Enojado": puntuacionEnojado,
        "Neutral": puntuacionNeutral
    }
    
    # La emoción con mayor puntuación gana
    emocion = max(puntuaciones.items(), key=lambda x: x[1])[0]
    
    # Agregar emoción detectada al historial
    historialEmociones.append(emocion)
    
    # Determinar la emoción predominante (la que más se repite en el historial)
    if historialEmociones:
        from collections import Counter
        emocionPredominante = Counter(historialEmociones).most_common(1)[0][0]
    else:
        emocionPredominante = emocion
    
    # Mostrar emoción en el frame
    colorEmocion = {
        "Feliz": (0, 255, 0),
        "Triste": (255, 0, 0),
        "Enojado": (0, 0, 255),
        "Neutral": (255, 255, 0)
    }.get(emocionPredominante, (255, 255, 255))
    
    cv2.putText(frame, f"Emoción: {emocionPredominante}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorEmocion, 2)
    
    # Mostrar puntuaciones para depuración
    cv2.putText(frame, f"F:{puntuacionFeliz} T:{puntuacionTriste} E:{puntuacionEnojado} N:{puntuacionNeutral}", (10, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return emocionPredominante

def main():
    """Función principal que ejecuta el programa."""
    global contadorParpadeos, ultimoTiempo
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Espejo para mayor naturalidad
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Si se detecta una cara
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujar la malla facial completa para visualización
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Extraer puntos específicos
                puntos = {}
                for idx in puntosFaciales:
                    if idx < len(face_landmarks.landmark):
                        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        puntos[idx] = (x, y)
                        # Dibujar puntos específicos
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
                # Análisis de persona viva
                esViva = detectarPersonaViva(puntos, frame)
                
                # Análisis de emoción
                emocion = detectarEmocion(puntos, frame)
                
                # Dibujar líneas entre puntos clave para mejor visualización
                # Ojos
                for i in range(len(puntosOjos)-1):
                    if puntosOjos[i] in puntos and puntosOjos[i+1] in puntos:
                        cv2.line(frame, puntos[puntosOjos[i]], puntos[puntosOjos[i+1]], (0, 255, 0), 1)
                
                # Boca
                for i in range(len(puntosBoca)-1):
                    if puntosBoca[i] in puntos and puntosBoca[i+1] in puntos:
                        cv2.line(frame, puntos[puntosBoca[i]], puntos[puntosBoca[i+1]], (0, 255, 0), 1)
                
                # También conectar el último punto con el primero para cerrar el contorno
                if puntosBoca[0] in puntos and puntosBoca[-1] in puntos:
                    cv2.line(frame, puntos[puntosBoca[-1]], puntos[puntosBoca[0]], (0, 255, 0), 1)

        # Mostrar la imagen con las detecciones
        cv2.imshow('PuntosFaciales y Emociones', frame)
        
        # Salir con la tecla q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()