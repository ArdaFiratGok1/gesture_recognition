import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib # Modeli kaydetmek için

# 1. Veri Setini Yükle
CSV_FILE = 'movements.csv'
MODEL_FILE = 'hareket_modeli.pkl'

data = pd.read_csv(CSV_FILE)

if data.empty:
    print(f"{CSV_FILE} dosyası boş veya bulunamadı.")
    print("Lütfen önce 1_veri_topla.py programını çalıştırarak veri toplayın.")
else:
    # 2. Veriyi Özellikler (X) ve Etiketler (y) olarak ayır
    # 'etiket' sütunu dışındaki tüm sütunlar özelliktir
    X = data.drop('etiket', axis=1) 
    y = data['etiket']

    # 3. Veriyi Eğitim ve Test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Modeli Oluştur ve Eğit (k-NN)
    # n_neighbors=3 -> en yakın 3 komşuya bak
    # Farklı etiket sayınız varsa (örn: 5 hareket) n_neighbors=5 deneyebilirsiniz
    model = KNeighborsClassifier(n_neighbors=3)
    
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