import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# MediaPipe el modelini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Kaydedilecek CSV dosyası
CSV_FILE = 'movements.csv'
# Etiket isimleri (Klavyede basılacak tuşlara göre)
# Örn: '0' tuşu -> yumruk, '1' tuşu -> açık el, '2' tuşu -> barış işareti
ETIKETLER = {
    '0': 'Yumruk',
    '1': 'Acik El',
    '2': 'Baris Isareti'
    # Buraya kendi hareketleriniz için daha fazla etiket ekleyebilirsiniz
}

# CSV dosyasını yazma modunda aç
# 21 kilit nokta var, her biri için (x, y) -> 42 özellik (feature)
# Artı bir de 'etiket' sütunu
features = []
for i in range(21):
    features.extend([f'x{i}', f'y{i}'])
features.append('etiket')

# Dosya yoksa başlık satırını yaz
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(features)

cap = cv2.VideoCapture(0)
print("Veri toplama programı başlatıldı.")
print("Veri kaydetmek için şu tuşlara basın:")
for key, value in ETIKETLER.items():
    print(f"  '{key}' tuşu -> {value}")
print("Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera okunamadı.")
        continue

    # Görüntüyü çevir (selfie modu) ve BGR'dan RGB'ye dönüştür
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # MediaPipe ile el tespiti
    results = hands.process(image)
    
    # Görüntüyü tekrar BGR'a dönüştür (OpenCV için)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # O an basılan tuşu yakala
    key = cv2.waitKey(5) & 0xFF
    
    # Çıkış için 'q'
    if key == ord('q'):
        break

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Eli ekrana çiz
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Veri kaydetme
            # ----------------------------------------------------
            # Eğer basılan tuş ETIKETLER içinde tanımlıysa
            if chr(key) in ETIKETLER:
                etiket_adi = ETIKETLER[chr(key)]
                print(f"Kaydediliyor... {etiket_adi}")
                
                # Normalizasyon ve Veri Çıkarma
                landmarks_list = []
                # Bilek noktasını (landmark 0) al
                wrist_x = hand_landmarks.landmark[0].x
                wrist_y = hand_landmarks.landmark[0].y

                for landmark in hand_landmarks.landmark:
                    # Tüm noktaların koordinatlarından bilek koordinatını çıkar
                    normalized_x = landmark.x - wrist_x
                    normalized_y = landmark.y - wrist_y
                    landmarks_list.extend([normalized_x, normalized_y])
                
                # Etiketi de listeye ekle
                landmarks_list.append(etiket_adi)
                
                # CSV dosyasına bu satırı ekle
                with open(CSV_FILE, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(landmarks_list)
            # ----------------------------------------------------

    cv2.imshow('Veri Toplama (cikis icin q)', image)

hands.close()
cap.release()
cv2.destroyAllWindows()