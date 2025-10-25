import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib # Modeli kaydetmek için
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# sklearn'den gelen spesifik "feature names" uyarısını gizle
warnings.filterwarnings("ignore", category=UserWarning)


# 1. Veri Setini Yükle
CSV_FILE = 'hareketler_ve_pozlar.csv' # Yeni CSV dosyanın adı
MODEL_FILE = 'hareket_ve_poz_modeli.pkl' # Eğitilmiş modelin kaydedileceği yeni dosya adı

if not os.path.exists(CSV_FILE):
    print(f"HATA: '{CSV_FILE}' dosyası bulunamadı.")
    print("Lütfen önce 1_veri_topla.py programını çalıştırarak veri toplayın.")
    exit()
    
data = pd.read_csv(CSV_FILE)

if data.empty:
    print(f"{CSV_FILE} dosyası boş.")
    print("Lütfen 1_veri_topla.py programını çalıştırarak veri toplayın.")
else:
    # 2. Veriyi Özellikler (X) ve Etiketler (y) olarak ayır
    X = data.drop('etiket', axis=1) 
    y = data['etiket']

    # Yeterli veri kontrolü (her etiketten en az 2 örnek olmalı)
    if y.nunique() > 1 and y.groupby(y).size().min() < 2:
        print(f"HATA: '{CSV_FILE}' dosyasındaki bazı etiketlerin sadece 1 örneği var.")
        print("Lütfen 1_veri_topla.py programını çalıştırarak her etiketten en az 2 adet (tercihen 50+) veri toplayın.")
        exit()

    # 3. Veriyi Eğitim ve Test setlerine ayır
    # (stratify=y, veriyi etiketlere göre orantılı ayırır, bu çok önemli)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Modeli Oluştur ve Eğit (k-NN)
    model = KNeighborsClassifier(n_neighbors=5)
    
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    print("Model eğitildi.")

    # 5. Modeli Test Et
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Modelin Test Doğruluğu: {accuracy * 100:.2f}%")

    # 6. Eğitilmiş Modeli Kaydet
    joblib.dump(model, MODEL_FILE)
    print(f"Model '{MODEL_FILE}' dosyasına kaydedildi.")