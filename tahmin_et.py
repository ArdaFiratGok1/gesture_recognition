import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# MediaPipe el ve yüz modelini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# !!! DEĞİŞİKLİK: max_num_hands=2 !!!
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eğitilmiş modeli yükle
MODEL_FILE = 'iki_elli_model.pkl' # !!! YENİ MODEL ADI !!!
if not os.path.exists(MODEL_FILE):
    print(f"HATA: '{MODEL_FILE}' modeli bulunamadı.")
    exit()

model = joblib.load(MODEL_FILE)

# --- Yüz kilit noktası index'leri (Bu kısım aynı) ---
face_landmark_indices = [
    1,   # Burun ucu (referans noktası)
    61, 291, 0, 39, 269, 13, 14, 17
]

cap = cv2.VideoCapture(0)
mevcut_hareket = None
hareket_degisti = False
print("Tahmin programı (İki El - Farazi Mod) başlatıldı. Çıkış için 'q' tuşuna basın.")

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
    
    tahmin_edilen_hareket = "" 

    # Çizim (İsteğe bağlı)
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

    # --- Veri Çıkarma ve Normalizasyon (1_veri_topla.py ile BİREBİR AYNI) ---
    
    nose_tip_x, nose_tip_y = None, None
    if face_results.multi_face_landmarks:
        nose_tip = face_results.multi_face_landmarks[0].landmark[1] 
        nose_tip_x = nose_tip.x
        nose_tip_y = nose_tip.y
    
    if nose_tip_x is None:
        cv2.putText(image, 'Yuz tespit edilemedi', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Gercek Zamanli Tahmin (cikis icin q)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue 

    # --- İKİ EL İŞLEME LOGIĞI ---
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

    # --- Yüz Verisi İşleme ---
    yuz_verisi = [0.0] * (len(face_landmark_indices) * 2)
    if face_results.multi_face_landmarks:
        temp_face_list = []
        for idx in face_landmark_indices:
            face_point = face_results.multi_face_landmarks[0].landmark[idx]
            normalized_x = face_point.x - nose_tip_x
            normalized_y = face_point.y - nose_tip_y
            temp_face_list.extend([normalized_x, normalized_y])
        yuz_verisi = temp_face_list

    # --- Tüm Veriyi Birleştir ve Tahmin Et ---
    landmarks_list = []
    landmarks_list.extend(sol_el_verisi)
    landmarks_list.extend(sag_el_verisi)
    landmarks_list.extend(yuz_verisi)

    data_row = np.array(landmarks_list).reshape(1, -1)
    
    # --- Benzerlik Eşiği Yöntemi (Önerilir) ---
    BENZERLIK_ESIGI = 0.6  # 5 komşudan en az 3'ü aynı olmalı

    probabilities = model.predict_proba(data_row)[0]
    max_prob = np.max(probabilities)
    
    if max_prob >= BENZERLIK_ESIGI:
        tahmin_index = np.argmax(probabilities)
        tahmin_edilen_hareket = model.classes_[tahmin_index]
    else:
        tahmin_edilen_hareket = "Belirsiz" # Veya "Bos" etiketin neyse o

    # --- Aksiyon Tetikleme ---
    cv2.putText(image, f'Tahmin: {tahmin_edilen_hareket}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if tahmin_edilen_hareket != "Belirsiz" and tahmin_edilen_hareket != mevcut_hareket:
        mevcut_hareket = tahmin_edilen_hareket
        hareket_degisti = True
    elif tahmin_edilen_hareket == "Belirsiz":
        mevcut_hareket = "Belirsiz"
        hareket_degisti = False
        
    if hareket_degisti:
        print(f"---------------------------------")
        print(f"AKSIYON TETIKLENDI: {mevcut_hareket}")
        print(f"---------------------------------")
        hareket_degisti = False 

    cv2.imshow('Gercek Zamanli Tahmin (Iki El - cikis icin q)', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()