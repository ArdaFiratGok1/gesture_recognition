import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import warnings
import pygame # <<< SES İÇİN EKLENDİ

# ... (tüm import satırlarından sonra) ...

def resize_with_padding(img, target_width, target_height, color=(0, 0, 0)):
    """
    Bir görüntüyü en-boy oranını koruyarak hedeflenen boyuta sığdırır
    ve boşlukları siyah renkle (veya 'color' ile) doldurur.
    """
    original_height, original_width = img.shape[:2]
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        # Görüntü hedeften daha geniş, genişliğe sığdır
        new_width = target_width
        new_height = int(new_width / original_ratio)
    else:
        # Görüntü hedeften daha uzun (veya aynı oranda), yüksekliğe sığdır
        new_height = target_height
        new_width = int(new_height * original_ratio)

    # Görüntüyü yeniden boyutlandır
    try:
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"HATA: cv2.resize başarısız oldu: {e}")
        # Hata durumunda boş bir resim döndür
        return np.full((target_height, target_width, 3), color, dtype=np.uint8)


    # Siyah bir tuval oluştur
    canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)

    # Yeniden boyutlandırılmış görüntüyü tuvalin ortasına yapıştır
    y_center = (target_height - new_height) // 2
    x_center = (target_width - new_width) // 2

    canvas[y_center:y_center + new_height, x_center:x_center + new_width] = resized_img

    return canvas

# --- Pygame mixer'ı başlat (SES İÇİN) ---
pygame.mixer.init()
# ... (kodun geri kalanı buradan devam eder) ...

# sklearn uyarılarını gizle
warnings.filterwarnings("ignore", category=UserWarning)

# --- Pygame mixer'ı başlat (SES İÇİN) ---
pygame.mixer.init()

# ------------------------------------------------------------------
# !!! EN ÖNEMLİ KISIM: BURAYI KENDİNE GÖRE DÜZENLE !!!
# ------------------------------------------------------------------
# '1_veri_topla.py' dosyasında kullandığın etiketlerle
# çalınacak ses ve gösterilecek resim dosyalarını eşleştir.
# Dosyaların bu script ile aynı klasörde (C:\gesture_ml) olduğundan emin ol.
AKSIYONLAR = {
    'neutral': {
        'ses': None, # Boş poz için ses/resim istemeyebiliriz
        'görüntü': 'images/neutral_monke.webp'
    },
    'thinking': {
        'ses': None,      # Kendi ses dosyanın adı
        'görüntü': 'images/thinking_monke.webp' # Kendi resim dosyanın adı
    },
    'finding': {
        'ses': None,
        'görüntü': 'images/finding_monke.jpeg'
    },
    'scared': {
        'ses': None,
        'görüntü': 'images/scared_monke.webp'
    }
    # Buraya '1_veri_topla.py' dosyasındaki TÜM etiketlerini eklemelisin
}
# ------------------------------------------------------------------

# --- Ses dosyalarını yükle (Hata kontrolü ile) ---
sounds = {}
for etiket, aksiyon in AKSIYONLAR.items():
    if 'ses' in aksiyon and aksiyon['ses'] and os.path.exists(aksiyon['ses']):
        try:
            sounds[etiket] = pygame.mixer.Sound(aksiyon['ses'])
        except pygame.error as e:
            print(f"HATA: '{aksiyon['ses']}' ses dosyası yüklenemedi. {e}")
    elif 'ses' in aksiyon and aksiyon['ses']:
        print(f"Uyarı: '{etiket}' için ses dosyası bulunamadı: {aksiyon['ses']}")

# --- Görüntü dosyalarını yükle (Hata kontrolü ile) ---
# --- Görüntü dosyalarını yükle (Hata kontrolü ile) ---

# !!! YENİ: Tüm aksiyon pencereleri için standart boyut belirle
AKSIYON_GENISLIK = 800
AKSIYON_YUKSEKLIK = 600

images = {}
for etiket, aksiyon in AKSIYONLAR.items():
    if 'görüntü' in aksiyon and aksiyon['görüntü'] and os.path.exists(aksiyon['görüntü']):
        # !!! DEĞİŞİKLİK: Resmi oku
        original_image = cv2.imread(aksiyon['görüntü'])
        
        # !!! DEĞİŞİKLİK: Resmi standart boyuta getir ve kaydet
        images[etiket] = resize_with_padding(original_image, AKSIYON_GENISLIK, AKSIYON_YUKSEKLIK)
        
    elif 'görüntü' in aksiyon and aksiyon['görüntü']:
        print(f"Uyarı: '{etiket}' için görüntü dosyası bulunamadı: {aksiyon['görüntü']}")


# --- MediaPipe Modellerini Başlat (İki El) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Eğitilmiş Modeli Yükle ---
MODEL_FILE = 'iki_elli_model.pkl' # İki elli modelin adı
if not os.path.exists(MODEL_FILE):
    print(f"HATA: '{MODEL_FILE}' modeli bulunamadı.")
    exit()

model = joblib.load(MODEL_FILE)

# Yüz kilit noktası index'leri (1_veri_topla.py ile aynı olmalı)
face_landmark_indices = [
    1,   # Burun ucu (referans noktası)
    61, 291, 0, 39, 269, 13, 14, 17
]

cap = cv2.VideoCapture(0)
mevcut_hareket = None
hareket_degisti = False
print("Tahmin programı (Aksiyonlu Mod) başlatıldı. Çıkış için 'q' tuşuna basın.")

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

    sol_el_verisi = [0.0] * (21 * 2)
    sag_el_verisi = [0.0] * (21 * 2)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            el_etiketi = handedness.classification[0].label
            temp_hand_list = []
            for landmark in hand_landmarks.landmark:
                temp_hand_list.extend([landmark.x - nose_tip_x, landmark.y - nose_tip_y])
            
            if el_etiketi == 'Left':
                sol_el_verisi = temp_hand_list
            elif el_etiketi == 'Right':
                sag_el_verisi = temp_hand_list

    yuz_verisi = [0.0] * (len(face_landmark_indices) * 2)
    if face_results.multi_face_landmarks:
        temp_face_list = []
        for idx in face_landmark_indices:
            face_point = face_results.multi_face_landmarks[0].landmark[idx]
            temp_face_list.extend([face_point.x - nose_tip_x, face_point.y - nose_tip_y])
        yuz_verisi = temp_face_list

    landmarks_list = []
    landmarks_list.extend(sol_el_verisi)
    landmarks_list.extend(sag_el_verisi)
    landmarks_list.extend(yuz_verisi)

    data_row = np.array(landmarks_list).reshape(1, -1)
    
    # --- Benzerlik Eşiği ile Tahmin ---
    BENZERLIK_ESIGI = 0.6 
    probabilities = model.predict_proba(data_row)[0]
    max_prob = np.max(probabilities)
    
    if max_prob >= BENZERLIK_ESIGI:
        tahmin_index = np.argmax(probabilities)
        tahmin_edilen_hareket = model.classes_[tahmin_index]
    else:
        # Eğer 'Bos' diye bir sınıf öğrettiysen, "Belirsiz" yerine onu kullanmak daha iyi olabilir
        tahmin_edilen_hareket = "Bos" # Veya "Belirsiz"

    # Tahmini ekrana yaz
    cv2.putText(image, f'Tahmin: {tahmin_edilen_hareket}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # --- AKSİYON TETİKLEME (GERÇEK) ---
    # Sadece hareket değiştiği anda aksiyonu tetikle
    if tahmin_edilen_hareket and tahmin_edilen_hareket != mevcut_hareket:
        mevcut_hareket = tahmin_edilen_hareket
        hareket_degisti = True
        print(f"Yeni hareket tespit edildi: {mevcut_hareket}") # Terminale bilgi ver
    elif not tahmin_edilen_hareket:
        mevcut_hareket = None
        hareket_degisti = False
        
    if hareket_degisti:
        # Ses çal
        if mevcut_hareket in sounds:
            pygame.mixer.stop() # Önceki sesi durdur
            sounds[mevcut_hareket].play()
            
        # Görüntü göster
        if mevcut_hareket in images:
            # Görüntüyü 'Aksiyon' adlı yeni bir pencerede göster
            cv2.imshow('Aksiyon', images[mevcut_hareket])
        else:
            # Eğer o hareket için resim yoksa ('Bos' pozu gibi), 'Aksiyon' penceresini kapat
            # Pencere hiç açılmadıysa getWindowProperty hata verir, try/except kullanalım.
            try:
                # 'Aksiyon' penceresini kapatmayı dene
                cv2.destroyWindow('Aksiyon')
            except cv2.error:
                # Eğer pencere zaten yoksa (veya hiç açılmadıysa) hata verir,
                # bu hatayı görmezden gel ve devam et.
                pass

        hareket_degisti = False # Eylemi yaptık, tekrar bekle

    # --- Çizim (İsteğe bağlı, en sonda olması daha iyi) ---
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

    cv2.imshow('Gercek Zamanli Tahmin (cikis icin q)', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Temizlik
hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit() # Pygame'i kapat



