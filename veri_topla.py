import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# MediaPipe el ve yüz modelini başlat
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kaydedilecek CSV dosyası
CSV_FILE = 'hareketler_ve_pozlar.csv' # Yeni bir CSV adı verelim

# Etiket isimleri
ETIKETLER = {
    '0': 'El Serbest',      # El normal duruyor
    '1': 'Yumruk_Yuzden_Uzak', # Yumruk, yüzden uzakta
    '2': 'Acik_El_Yanakta',    # Açık el yanakta
    '3': 'Parmak_Dudakta',     # İşaret parmağı dudakta
    '4': 'Dil_Disarida'        # Dil dışarıda
    # Buraya yeni pozlar ve el hareketleri ekleyebilirsin
}

# CSV başlıkları
features = []
# El kilit noktaları (21 adet, her biri için x, y)
for i in range(21):
    features.extend([f'hand_x{i}', f'hand_y{i}'])

# Yüz kilit noktaları (Sadece belirli noktaları veya daha azını seçebiliriz)
# MediaPipe Face Mesh'te 468 nokta var, hepsini almak çok fazla olabilir.
# Dudaklar, burun, çene gibi kritik noktaları seçelim.
# Bunlar, MediaPipe Face Mesh'in dokümantasyonundan alınmış örnekler
# (Tam listeyi ve index'leri kontrol etmek için MediaPipe dokümantasyonuna bakılmalı)
# Şimdilik örnek olarak bazı kritik noktaları alalım: Burun ucu, dudak köşeleri, çene ucu
face_landmark_indices = [
    1,   # Burun ucu (referans noktası olarak kullanacağız)
    61, 291, # Dudak köşeleri (sol ve sağ)
    0,   # Çene ucu
    39, 269, # Sağ göz dış köşesi, sol göz dış köşesi
    13, 14, # Üst ve Alt dudak orta noktası (parmak dudağa değme için)
    17 # Alt dudak altı (dil için)
]
for i in face_landmark_indices:
    features.extend([f'face_x{i}', f'face_y{i}'])

features.append('etiket') # Son sütun etiket olacak

# Dosya yoksa başlık satırını yaz
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(features)

cap = cv2.VideoCapture(0)
print("Veri toplama programı başlatıldı.")
print(f"Veri '{CSV_FILE}' dosyasına kaydedilecek.")
print("Veri kaydetmek için şu tuşlara basın:")
for key, value in ETIKETLER.items():
    print(f"   '{key}' tuşu -> {value}")
print("Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Kamera okunamadı.")
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Performans için görüntüyü yazılabilir değil olarak işaretle
    image.flags.writeable = False
    
    # El ve yüz tespiti
    hand_results = hands.process(image)
    face_results = face_mesh.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q'):
        break

    # Kilit noktalarını çiz
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Sadece seçtiğimiz face_landmark_indices noktalarını çizmek istersek:
            for idx in face_landmark_indices:
                if 0 <= idx < len(face_landmarks.landmark):
                    point = face_landmarks.landmark[idx]
                    h, w, c = image.shape
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.circle(image, (cx, cy), 2, (0, 255, 0), -1) # Yeşil noktalar
            # Tüm yüz ağını çizmek için bu satırı kullanabilirsin (çok yoğun olur):
            # mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)


    # --- Veri Kaydetme ---
    if chr(key) in ETIKETLER:
        etiket_adi = ETIKETLER[chr(key)]
        print(f"Kaydediliyor... {etiket_adi}")
        
        landmarks_list = []
        
        # --- Normalizasyon: Referans noktası BURUN UCUNUN MERKEZİ ---
        # Burun ucu kilit noktasının indeksi (face_landmark_indices listesinden alınır)
        # Bu genelde MediaPipe Face Mesh'te #1 numaralı kilit noktadır.
        # Eğer yüz tespit edilmediyse bu kısım sorun çıkarır, kontrol edelim
        
        nose_tip_x, nose_tip_y = None, None
        if face_results.multi_face_landmarks:
            # İlk yüzün burun ucu
            nose_tip = face_results.multi_face_landmarks[0].landmark[1] # MediaPipe Face Mesh'te burun ucu genellikle 1 numaralı noktadır
            nose_tip_x = nose_tip.x
            nose_tip_y = nose_tip.y
        
        # Eğer burun ucu tespit edilemediyse veri kaydetme
        if nose_tip_x is None or nose_tip_y is None:
            print("Uyarı: Yüz tespit edilemedi, veri kaydedilemiyor.")
            continue # Bu frame'i atla
        
        # EL KİLİT NOKTALARI (Burun ucuna göre normalize edilmiş)
        if hand_results.multi_hand_landmarks:
            for hand_landmark_in_frame in hand_results.multi_hand_landmarks:
                for landmark in hand_landmark_in_frame.landmark:
                    normalized_x = landmark.x - nose_tip_x
                    normalized_y = landmark.y - nose_tip_y
                    landmarks_list.extend([normalized_x, normalized_y])
        else:
            # El tespit edilemediyse, el verisi yerine sıfırlarla doldur
            # 21 el noktası var, her biri için x, y => 42 sıfır
            landmarks_list.extend([0.0] * (21 * 2))

        # YÜZ KİLİT NOKTALARI (Burun ucuna göre normalize edilmiş)
        if face_results.multi_face_landmarks:
            for idx in face_landmark_indices: # Sadece seçtiğimiz yüz noktalarını al
                face_point = face_results.multi_face_landmarks[0].landmark[idx]
                normalized_x = face_point.x - nose_tip_x
                normalized_y = face_point.y - nose_tip_y
                landmarks_list.extend([normalized_x, normalized_y])
        else:
            # Yüz tespit edilemediyse, yüz verisi yerine sıfırlarla doldur
            # face_landmark_indices listesindeki her nokta için x, y => len(face_landmark_indices) * 2 sıfır
            landmarks_list.extend([0.0] * (len(face_landmark_indices) * 2))

        # Etiketi de listeye ekle
        landmarks_list.append(etiket_adi)
        
        # CSV dosyasına bu satırı ekle
        with open(CSV_FILE, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks_list)

    cv2.imshow('Veri Toplama (cikis icin q)', image)

hands.close()
face_mesh.close() # Yeni eklenen yüz modelini kapat
cap.release()
cv2.destroyAllWindows()