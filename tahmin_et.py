import cv2
import mediapipe as mp
import joblib # Modeli yüklemek için
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# sklearn'den gelen spesifik "feature names" uyarısını gizle
warnings.filterwarnings("ignore", category=UserWarning)


# MediaPipe el ve yüz modelini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Eğitilmiş modeli yükle
MODEL_FILE = 'hareket_ve_poz_modeli.pkl' # Yeni model dosyasının adı
if not os.path.exists(MODEL_FILE):
    print(f"HATA: '{MODEL_FILE}' modeli bulunamadı.")
    print("Lütfen önce 2_model_egit.py programını çalıştırın.")
    exit()

model = joblib.load(MODEL_FILE)

cap = cv2.VideoCapture(0)

mevcut_hareket = None
hareket_degisti = False

# Yüz kilit noktaları için index'leri tekrar tanımla
face_landmark_indices = [
    1,   # Burun ucu (referans noktası olarak kullanacağız)
    61, 291, # Dudak köşeleri (sol ve sağ)
    0,   # Çene ucu
    39, 269, # Sağ göz dış köşesi, sol göz dış köşesi
    13, 14, # Üst ve Alt dudak orta noktası (parmak dudağa değme için)
    17 # Alt dudak altı (dil için)
]

print("Tahmin programı (Farazi Mod) başlatıldı. Çıkış için 'q' tuşuna basın.")

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

    # Kilit noktalarını çiz (optiysenel)
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

    # --- Veri Çıkarma ve Normalizasyon ---
    landmarks_list = []
    
    nose_tip_x, nose_tip_y = None, None
    if face_results.multi_face_landmarks:
        nose_tip = face_results.multi_face_landmarks[0].landmark[1] 
        nose_tip_x = nose_tip.x
        nose_tip_y = nose_tip.y
    
    if nose_tip_x is None or nose_tip_y is None:
        cv2.putText(image, 'Yuz tespit edilemedi', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Gercek Zamanli Tahmin (cikis icin q)', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue # Yüz yoksa tahmine geçme

    # EL KİLİT NOKTALARI (Burun ucuna göre normalize edilmiş)
    if hand_results.multi_hand_landmarks:
        for hand_landmark_in_frame in hand_results.multi_hand_landmarks:
            for landmark in hand_landmark_in_frame.landmark:
                normalized_x = landmark.x - nose_tip_x
                normalized_y = landmark.y - nose_tip_y
                landmarks_list.extend([normalized_x, normalized_y])
    else:
        landmarks_list.extend([0.0] * (21 * 2))

    # YÜZ KİLİT NOKTALARI (Burun ucuna göre normalize edilmiş)
    if face_results.multi_face_landmarks:
        for idx in face_landmark_indices:
            face_point = face_results.multi_face_landmarks[0].landmark[idx]
            normalized_x = face_point.x - nose_tip_x
            normalized_y = face_point.y - nose_tip_y
            landmarks_list.extend([normalized_x, normalized_y])
    else:
        landmarks_list.extend([0.0] * (len(face_landmark_indices) * 2))

    # Modeli kullanarak tahmin yap
    data_row = np.array(landmarks_list).reshape(1, -1)
    tahmin = model.predict(data_row)
    tahmin_edilen_hareket = tahmin[0]

    # Tahmin edilen hareketi ekrana yazdır
    cv2.putText(image, f'Tahmin: {tahmin_edilen_hareket}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # --- Aksiyon Tetikleme ---
    if tahmin_edilen_hareket and tahmin_edilen_hareket != mevcut_hareket:
        mevcut_hareket = tahmin_edilen_hareket
        hareket_degisti = True
    elif not tahmin_edilen_hareket: # Eğer el/yüz kaybolursa
        mevcut_hareket = None
        hareket_degisti = False
        
    if hareket_degisti:
        print(f"---------------------------------")
        print(f"AKSIYON TETIKLENDI: {mevcut_hareket}")
        print(f"---------------------------------")
        hareket_degisti = False 

    cv2.imshow('Gercek Zamanli Tahmin (cikis icin q)', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()