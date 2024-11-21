import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import pandas as pd
from datetime import datetime

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Carregar o modelo YOLO11
model = YOLO("yolo11s.pt")
names = model.model.names

# Iniciar captura de vídeo (aqui usando webcam)
cap = cv2.VideoCapture(0)
count = 0
inside_room_count = 0  # Contagem de pessoas dentro da sala
track_memory = {}
inside_room_ids = set()  # IDs de pessoas atualmente dentro da sala

# Definir a posição e tamanho do retângulo que representa a sala para ocupar 2/3 da tela
frame_width, frame_height = 1020, 600
room_top_left = (frame_width // 6, frame_height // 6)  # 1/6 de margem em cada lado
room_bottom_right = (5 * frame_width // 6, 5 * frame_height // 6)  # 5/6 da largura e altura


# Lista para armazenar dados das entradas e saídas
data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    
    # Executar rastreamento com YOLO11 no quadro atual
    results = model.track(frame, persist=True, classes=0)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Obter boxes, IDs de classe, IDs de rastreamento e scores de confiança
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Centro do box
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # Exibir a caixa delimitadora e identificador de rastreamento
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

            # Verificar e atualizar contagem de entrada
            if track_id not in track_memory:
                track_memory[track_id] = (cx, cy)

            prev_position = track_memory[track_id]

            # Verificar se a pessoa entrou na sala
            if track_id not in inside_room_ids and room_top_left[0] <= cx <= room_bottom_right[0] and room_top_left[1] <= cy <= room_bottom_right[1]:
                inside_room_ids.add(track_id)  # Marcar ID como dentro da sala
                data.append({
                    "ID": track_id,
                    "Entrada": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Saída": None  # A saída será registrada mais tarde
                })
                inside_room_count += 1
                print(f"Pessoa ID {track_id} entrou na sala.")

            # Verificar se a pessoa saiu da sala
            elif track_id in inside_room_ids and (cx < room_top_left[0] or cx > room_bottom_right[0] or cy < room_top_left[1] or cy > room_bottom_right[1]):
                # Encontrar a entrada correspondente para essa pessoa e registrar a saída
                for entry in data:
                    if entry["ID"] == track_id and entry["Saída"] is None:
                        entry["Saída"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Pessoa ID {track_id} saiu da sala.")
                inside_room_ids.remove(track_id)  # Remover o ID da lista de pessoas dentro

            # Atualizar a posição atual
            track_memory[track_id] = (cx, cy)

    # Desenhar o quadrado que representa a sala
    cv2.rectangle(frame, room_top_left, room_bottom_right, (255, 0, 0), 2)

    # Exibir contagem de entradas
    cvzone.putTextRect(frame, f'Contagem de entradas: {inside_room_count}', (50, 100), 2, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Salvar os dados no arquivo CSV
df = pd.DataFrame(data)
df.to_csv('entrada_saida_pessoas.csv', index=False)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
