import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

#! Para detectar movimiento necesitamos definir un ubral y hacer un seguimiento del historial de los puntos
UMBRAL_MOVIMIENTO = 2.5
HISTORIAL_FRAMES = 5    

# Lista de índices de landmarks específicos
selected_points = [
    33, 133, 160, 144, 145, 153,  # Ojo izquierdo
    362, 263, 385, 380, 374, 386,  # Ojo derecho
    # Boca
    61, 291, 0, 17, 40, 37,  # Labios externos
    78, 95, 88, 178, 87, 14,  # Labios internos
    # Cejas
    70, 63, 105, 66, 107, 55,  # Ceja izquierda
    336, 296, 334, 293, 300, 276  # Ceja derecha
] 

#! Para guardar el historial de movimiento
historialMovimiento = deque(maxlen=HISTORIAL_FRAMES)

#! Calculos de relación de aspecto
def calcularRelacionAspectoOjo(puntos):
    vertical1 = distancia(puntos[33], puntos[160])
    vertical2 = distancia(puntos[144], puntos[145])
    horizontal = distancia(puntos[133], puntos[33])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def calcularAperturaBoca(puntos):
    vertical = distancia(puntos[61], puntos[291])
    horizontal = distancia(puntos[0], puntos[17])
    return vertical / horizontal

def detectarVida(variaciones):
    # Analiza el historial de movimientos
    if len(variaciones) < HISTORIAL_FRAMES:
        return False
    return np.mean(variaciones) > UMBRAL_MOVIMIENTO

EMOCIONES = {
    'neutral': {'cejas': 0.15, 'boca': 0.3},
    'feliz': {'cejas': 0.25, 'boca': 0.5},
    'triste': {'cejas': 0.1, 'boca': 0.2},
    'enojado': {'cejas': 0.3, 'boca': 0.4}
}

def analizarEmocion(ear, mar, cejas):
    if mar > EMOCIONES['feliz']['boca'] and cejas > EMOCIONES['feliz']['cejas']:
        return 'feliz'
    elif mar < EMOCIONES['triste']['boca'] and cejas < EMOCIONES['triste']['cejas']:
        return 'triste'
    elif mar > EMOCIONES['enojado']['boca'] and cejas > EMOCIONES['enojado']['cejas']:
        return 'enojado'
    return 'neutral'

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(rgb_frame)

    variacionActual = 0
    emocionActual = 'neutral'
    esPersonaViva = False

    if resultados.multi_face_landmarks:
        for landmarks in resultados.multi_face_landmarks:
            puntosFaciales = {}
            
            for idx in selected_points:
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                puntosFaciales[idx] = (x, y)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Cálculo de parámetros vitales
            earOjoIzq = calcularRelacionAspectoOjo([puntosFaciales[i] for i in [33, 133, 160, 144, 145, 153]])
            earOjoDer = calcularRelacionAspectoOjo([puntosFaciales[i] for i in [362, 263, 385, 380, 374, 386]])
            mar = calcularAperturaBoca(puntosFaciales)
            
            # Cálculo movimiento cejas
            alturaCejas = np.mean([
                distancia(puntosFaciales[70], puntosFaciales[55]),
                distancia(puntosFaciales[336], puntosFaciales[276])
            ])

            # Detección de variación
            if len(historialMovimiento) > 0:
                variacionActual = np.abs(np.mean(list(historialMovimiento)[-1]) - (earOjoIzq + earOjoDer))
            historialMovimiento.append((earOjoIzq, earOjoDer, mar))

            # Determinación de vida y emoción
            esPersonaViva = detectarVida(historialMovimiento)
            emocionActual = analizarEmocion((earOjoIzq + earOjoDer)/2, mar, alturaCejas)

    # Mostrar resultados
    colorEstado = (0, 255, 0) if esPersonaViva else (0, 0, 255)
    textoEstado = f'Persona viva: {"SI" if esPersonaViva else "NO"} - Emocion: {emocionActual}'
    
    cv2.putText(frame, textoEstado, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, colorEstado, 2)
    
    cv2.imshow('Deteccion Facial Avanzada', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()