import sys
import os
import cv2
import mediapipe as mp
import joblib
import numpy as np
import warnings
import pygame
import tkinter as tk
from tkinter import messagebox

import sklearn.neighbors
import sklearn.ensemble

def show_error_popup(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception as e:
        print(f"POPUP HATA: {e}")

if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_path = os.getcwd()

def resize_with_padding(img, target_width, target_height, color=(0, 0, 0)):
    original_height, original_width = img.shape[:2]
    if original_height == 0 or original_width == 0:
        return np.full((target_height, target_width, 3), color, dtype=np.uint8)
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    if original_ratio > target_ratio:
        new_width = target_width
        new_height = int(new_width / original_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * original_ratio)
    try:
        if new_width > 0 and new_height > 0:
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
             return np.full((target_height, target_width, 3), color, dtype=np.uint8)
    except Exception as e:
        print(f"HATA: cv2.resize başarısız oldu: {e}")
        return np.full((target_height, target_width, 3), color, dtype=np.uint8)
    canvas = np.full((target_height, target_width, 3), color, dtype=np.uint8)
    y_center = (target_height - new_height) // 2
    x_center = (target_width - new_width) // 2
    canvas[y_center:y_center + new_height, x_center:x_center + new_width] = resized_img
    return canvas

pygame.mixer.init()

warnings.filterwarnings("ignore", category=UserWarning)

AKSIYONLAR = {
    'neutral': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'neutral_monke.webp')
    },
    'thinking': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'thinking_monke.webp')
    },
    'finding': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'finding_monke.jpeg')
    },
    'scared': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'scared_monke.webp')
    },
    'surprised': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'holy_what_face.jpeg')
    },
    'sad': {
        'ses': None,
        'görüntü': os.path.join(base_path, 'images', 'sad_face.jpg')
    }
}
sounds = {}
for etiket, aksiyon in AKSIYONLAR.items():
    if 'ses' in aksiyon and aksiyon['ses'] and os.path.exists(aksiyon['ses']):
        try:
            sounds[etiket] = pygame.mixer.Sound(aksiyon['ses'])
        except pygame.error as e:
            print(f"HATA: '{aksiyon['ses']}' ses dosyası yüklenemedi. {e}")
    elif 'ses' in aksiyon and aksiyon['ses']:
        print(f"Uyarı: '{etiket}' için ses dosyası bulunamadı: {aksiyon['ses']}")

AKSIYON_GENISLIK = 800
AKSIYON_YUKSEKLIK = 600
images = {}
for etiket, aksiyon in AKSIYONLAR.items():
    if 'görüntü' in aksiyon and aksiyon['görüntü'] and os.path.exists(aksiyon['görüntü']):
        original_image = cv2.imread(aksiyon['görüntü'])
        if original_image is None:
            print(f"HATA: '{aksiyon['görüntü']}' dosyası okunamadı.")
            images[etiket] = np.full((AKSIYON_YUKSEKLIK, AKSIYON_GENISLIK, 3), (0,0,0), dtype=np.uint8)
        else:
            images[etiket] = resize_with_padding(original_image, AKSIYON_GENISLIK, AKSIYON_YUKSEKLIK)
    elif 'görüntü' in aksiyon and aksiyon['görüntü']:
        print(f"Uyarı: '{etiket}' için görüntü dosyası bulunamadı: {aksiyon['görüntü']}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

MODEL_FILE = os.path.join(base_path, 'iki_elli_model.pkl')
if not os.path.exists(MODEL_FILE):
    show_error_popup("Kritik Hata", f"Model dosyası bulunamadı!\n\nBeklenen yol: {MODEL_FILE}\n\nLütfen 'iki_elli_model.pkl' dosyasının .exe ile aynı klasörde olduğundan emin olun.")
    exit()

model = joblib.load(MODEL_FILE)

face_landmark_indices = [ 1, 61, 291, 0, 39, 269, 13, 14, 17 ]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    show_error_popup("Kritik Hata", "Kamera (0) açılamadı.\n\nLütfen kameranızın başka bir uygulama tarafından kullanılmadığından emin olun ve programı yeniden başlatın.")
    exit()
    
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

    nose_tip_x, nose_tip_y = None, None
    if face_results.multi_face_landmarks:
        nose_tip = face_results.multi_face_landmarks[0].landmark[1] 
        nose_tip_x = nose_tip.x
        nose_tip_y = nose_tip.y
    
    if nose_tip_x is None:
        cv2.putText(image, 'Yuz tespit edilemedi', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
    
    BENZERLIK_ESIGI = 0.6 
    probabilities = model.predict_proba(data_row)[0]
    max_prob = np.max(probabilities)
    tahmin_index = np.argmax(probabilities)
    tahmin_etiketi_en_yakin = model.classes_[tahmin_index]
    
    if max_prob >= BENZERLIK_ESIGI:
        tahmin_edilen_hareket = tahmin_etiketi_en_yakin
    else:
        tahmin_edilen_hareket = "neutral" 

    cv2.putText(image, f'Tahmin: {tahmin_edilen_hareket}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if tahmin_edilen_hareket and tahmin_edilen_hareket != mevcut_hareket:
        mevcut_hareket = tahmin_edilen_hareket
        hareket_degisti = True
        print(f"Yeni hareket tespit edildi: {mevcut_hareket}") 
    elif not tahmin_edilen_hareket:
        mevcut_hareket = None
        hareket_degisti = False
        
    if hareket_degisti:
        if mevcut_hareket in sounds:
            pygame.mixer.stop()
            sounds[mevcut_hareket].play()
        if mevcut_hareket in images:
            cv2.imshow('Aksiyon', images[mevcut_hareket])
        else:
            try:
                cv2.destroyWindow('Aksiyon')
            except cv2.error:
                pass
        hareket_degisti = False

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

hands.close()
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
