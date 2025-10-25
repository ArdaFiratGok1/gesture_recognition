import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import mediapipe as mp
import joblib # Modeli yüklemek için
import numpy as np
# import pygame # Şimdilik gerek yok
import os

# MediaPipe el modelini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Eğitilmiş modeli yükle
MODEL_FILE = 'hareket_modeli.pkl'
if not os.path.exists(MODEL_FILE):
    print(f"HATA: '{MODEL_FILE}' modeli bulunamadı.")
    print("Lütfen önce 2_model_egit.py programını çalıştırın.")
    exit()

model = joblib.load(MODEL_FILE)

# --- SES VE GÖRÜNTÜ KISIMLARI ŞİMDİLİK DEVRE DIŞI ---
# pygame.mixer.init()
# AKSIYONLAR = { ... }
# sounds = {}
# images = {}
# ---

cap = cv2.VideoCapture(0)

# Eylemlerin sürekli tetiklenmemesi için
mevcut_hareket = None
hareket_degisti = False

print("Tahmin programı (Farazi Mod) başlatıldı. Çıkış için 'q' tuşuna basın.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Görüntüyü çevir (selfie) ve RGB'ye dönüştür
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Performans için görüntüyü yazılabilir değil olarak işaretle
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    
    # Görüntüyü OpenCV için BGR'a geri dönüştür
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    tahmin_edilen_hareket = "" # Başlangıçta boş

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Eli ekrana çiz
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- Normalizasyon (Adım 1 ile %100 aynı olmalı!) ---
            landmarks_list = []
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y

            for landmark in hand_landmarks.landmark:
                normalized_x = landmark.x - wrist_x
                normalized_y = landmark.y - wrist_y
                landmarks_list.extend([normalized_x, normalized_y])
            # --- Normalizasyon Bitti ---
            
            # Modeli kullanarak tahmin yap
            data_row = np.array(landmarks_list).reshape(1, -1)
            tahmin = model.predict(data_row)
            tahmin_edilen_hareket = tahmin[0]

    # Tahmin edilen hareketi ekrana yazdır (En önemli kısım bu)
    cv2.putText(image, f'Tahmin: {tahmin_edilen_hareket}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # --- Aksiyon Tetikleme ---
    if tahmin_edilen_hareket and tahmin_edilen_hareket != mevcut_hareket:
        mevcut_hareket = tahmin_edilen_hareket
        hareket_degisti = True
    elif not tahmin_edilen_hareket:
        mevcut_hareket = None
        hareket_degisti = False
        
    # Sadece hareket değiştiği anda eylemi yap
    if hareket_degisti:
        
        # --- FARAZİ AKSİYON ---
        # Gerçek ses/görüntü yerine, terminale bir onay mesajı yazdıralım
        print(f"---------------------------------")
        print(f"AKSIYON TETIKLENDI: {mevcut_hareket}")
        print(f"---------------------------------")
        # --- Bitti ---

        hareket_degisti = False # Eylemi yaptık, tekrar bekle

    cv2.imshow('Gercek Zamanli Tahmin (cikis icin q)', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Temizlik
hands.close()
cap.release()
cv2.destroyAllWindows()