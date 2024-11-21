import face_recognition
import os
import sys
import cv2
import numpy as np
import math
from ultralytics import YOLO 
import cvzone 

def face_confidence(faceDistance, faceMatchThreshold=0.6):
    linearVal = 1.0 - faceDistance
    if faceDistance > faceMatchThreshold:
        return str(round(linearVal * 100, 2)) + '%'
    else:
        value = (linearVal + ((1.0 - linearVal) * math.pow(linearVal - 0.5, 2) * 0.2)) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.faceLocations = []
        self.faceEncodings = []
        self.faceNames = []
        self.knownFaceEncodings = []
        self.knownFaceNames = []
        self.unknown_face_counter = 0
        self.encode_faces()

    def encode_faces(self):
        self.knownFaceEncodings = []
        self.knownFaceNames = []
        
        faces_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faces')
        if not os.path.isdir(faces_dir):
            os.makedirs(faces_dir)

        for image in os.listdir(faces_dir):
            image_path = os.path.join(faces_dir, image)
            faceImage = face_recognition.load_image_file(image_path)
            faceEncodings = face_recognition.face_encodings(faceImage)

            if faceEncodings:
                self.knownFaceEncodings.append(faceEncodings[0])
                self.knownFaceNames.append(image)
        
        print("Imagens processadas:", self.knownFaceNames)

    def run_recognition(self):
        videoCapture = cv2.VideoCapture(0)

        if not videoCapture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = videoCapture.read()
            if not ret:
                print("Falha ao capturar o quadro da webcam.")
                break

            self.faceLocations = face_recognition.face_locations(frame)
            self.faceEncodings = face_recognition.face_encodings(frame, self.faceLocations)
            self.faceNames = []

            for faceEncoding in self.faceEncodings:
                name = 'Unknown'
                confidence = 'Unknown'

                if self.knownFaceEncodings:
                    matches = face_recognition.compare_faces(self.knownFaceEncodings, faceEncoding)
                    faceDistances = face_recognition.face_distance(self.knownFaceEncodings, faceEncoding)

                    if len(faceDistances) > 0:
                        bestMatchIndex = np.argmin(faceDistances)
                        if matches[bestMatchIndex]:
                            name = self.knownFaceNames[bestMatchIndex]
                            confidence = face_confidence(faceDistances[bestMatchIndex])

                # Salvar a face desconhecida se não corresponder a rostos conhecidos
                if name == 'Unknown':
                    if not self.is_duplicate(faceEncoding):
                        top, right, bottom, left = self.faceLocations[len(self.faceNames)]
                        self.save_unknown_face(frame, (top, right, bottom, left))
                        self.knownFaceEncodings.append(faceEncoding)  # Adiciona o novo encoding
                        self.knownFaceNames.append(f'pessoa_desconhecida_{self.unknown_face_counter - 1}')
                    else:
                        print("Rosto desconhecido já salvo, não será salvo novamente.")

                self.faceNames.append(name) 

            for (top, right, bottom, left), name in zip(self.faceLocations, self.faceNames):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        videoCapture.release()
        cv2.destroyAllWindows()

    def save_unknown_face(self, frame, face_location):
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        if face_image.size > 0:
            unknown_face_filename = os.path.join('faces', f'pessoa_desconhecida_{self.unknown_face_counter}.jpg')
            cv2.imwrite(unknown_face_filename, face_image)
            print(f'Rosto desconhecido salvo como: {unknown_face_filename}')
            self.unknown_face_counter += 1
        else:
            print("A imagem do rosto desconhecido está vazia e não será salva.")

    def is_duplicate(self, faceEncoding, threshold=0.6):
        matches = face_recognition.compare_faces(self.knownFaceEncodings, faceEncoding, tolerance=threshold)
        return any(matches)

# Função para carregar IDs de arquivos na pasta 'faces'
def load_ids_from_faces_folder(folder_path):
    ids = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Considera arquivos de imagem com extensão .jpg
            track_id = int(os.path.splitext(filename)[0])  # Extrair ID do nome do arquivo
            ids.add(track_id)
    return ids

if __name__ == '__main__':
    # Instanciar a classe de reconhecimento facial
    fr = FaceRecognition()

    # Inicializar inside_room_ids a partir dos arquivos da pasta 'faces'
    folder_path = 'faces'
    inside_room_ids = load_ids_from_faces_folder(folder_path)

    # Configurar o modelo YOLO11
    model = YOLO("yolo11s.pt")
    names = model.model.names

    # Definir retângulo que representa a área da sala
    frame_width, frame_height = 1020, 600
    room_top_left = (frame_width // 6, frame_height // 6)
    room_bottom_right = (5 * frame_width // 6, 5 * frame_height // 6)

    # Iniciar captura de vídeo
    cap = cv2.VideoCapture(0)
    track_memory = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Executar detecção com YOLO
        results = model.track(frame, persist=True, classes=0)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Obter caixas delimitadoras, IDs de classe e IDs de rastreamento
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centro do box

                # Verificar se o track_id detectado está na lista de IDs da pasta 'faces'
                if track_id in inside_room_ids:
                    # Verificar se a pessoa entrou na área da sala
                    if room_top_left[0] <= cx <= room_bottom_right[0] and room_top_left[1] <= cy <= room_bottom_right[1]:
                        # Chamar reconhecimento facial
                        fr.run_recognition(frame, track_id)

                # Exibir a caixa delimitadora e ID no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y2), 1, 1)

        # Desenhar o retângulo que representa a sala
        cv2.rectangle(frame, room_top_left, room_bottom_right, (255, 0, 0), 2)

        # Exibir a imagem
        cv2.imshow("Detecção de Pessoas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()