import face_recognition
import os
import sys
import cv2
import numpy as np
import math

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

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition() 
