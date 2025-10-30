import cv2
import mediapipe as mp
import csv
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

CSV_FILE = 'iki_elli_pozlar.csv'

ETIKETLER = {
    '0': 'neutral',
    '1': 'thinking',
    '2': 'finding',
    '3': 'scared' ,
    '4': 'sad' ,
    '5': 'surprised'
}

face_landmark_indices = [
    1,
    61, 291, 0, 39, 269, 13, 14, 17
]

features = []
for i in range(21):
    features.extend([f'sol_el_x{i}', f'sol_el_y{i}'])
for i in range(21):
    features.extend([f'sag_el_x{i}', f'sag_el_y{i}'])
for i in face_landmark_indices:
    features.extend([f'face_x{i}', f'face_y{i}'])

features.append('etiket')

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(features)

cap = cv2.VideoCapture(0)
print("Veri toplama programı (İki El) başlatıldı.")
print(f"Veri '{CSV_FILE}' dosyasına kaydedilecek.")
for key, value in ETIKETLER.items():
    print(f"    '{key}' tuşu -> {value}")
print("Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    hand_results = hands.process(image)
    face_results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx in face_landmark_indices:
                if 0 <= idx < len(face_landmarks.landmark):
                    point = face_landmarks.landmark[idx]
                    h, w, c = image.shape
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1)

    if chr(key) in ETIKETLER:
        etiket_adi = ETIKETLER[chr(key)]
        print(f"Kaydediliyor... {etiket_adi}")
        
        nose_tip_x, nose_tip_y = None, None
        if face_results.multi_face_landmarks:
            nose_tip = face_results.multi_face_landmarks[0].landmark[1]
            nose_tip_x = nose_tip.x
            nose_tip_y = nose_tip.y
        
        if nose_tip_x is None:
            print("Uyarı: Yüz tespit edilemedi, veri kaydedilemiyor.")
            continue
        
        sol_el_verisi = [0.0] * (21 * 2)
        sag_el_verisi = [0.0] * (21 * 2)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                el_etiketi = handedness.classification[0].label
                
                temp_hand_list = []
                for landmark in hand_landmarks.landmark:
                    normalized_x = landmark.x - nose_tip_x
                    normalized_y = landmark.y - nose_tip_y
                    temp_hand_list.extend([normalized_x, normalized_y])
                
                if el_etiketi == 'Left':
                    sol_el_verisi = temp_hand_list
                elif el_etiketi == 'Right':
                    sag_el_verisi = temp_hand_list

        yuz_verisi = [0.0] * (len(face_landmark_indices) * 2)
        if face_results.multi_face_landmarks:
            temp_face_list = []
            for idx in face_landmark_indices:
                face_point = face_results.multi_face_landmarks[0].landmark[idx]
                normalized_x = face_point.x - nose_tip_x
                normalized_y = face_point.y - nose_tip_y
                temp_face_list.extend([normalized_x, normalized_y])
            yuz_verisi = temp_face_list
        
        landmarks_list = []
        landmarks_list.extend(sol_el_verisi)
        landmarks_list.extend(sag_el_verisi)
        landmarks_list.extend(yuz_verisi)
        landmarks_list.append(etiket_adi)
        
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks_list)

    cv2.imshow('Veri Toplama (İki El - cikis icin q)', image)

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
