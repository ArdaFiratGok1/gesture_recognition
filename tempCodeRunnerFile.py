import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib 
import os
import warnings

# sklearn uyarılarını gizle
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Veri Setini Yükle
CSV_FILE = 'iki_elli_pozlar.csv' # !!! YENİ CSV ADI !!!
MODEL_FILE = 'iki_elli_model.pkl' # !!! YENİ MODEL ADI !!!

if not os.path.exists(CSV_FILE):
    print(f"HATA: '{CSV_FILE}' dosyası bulunamadı.")
    exit()
    
data = pd.read_csv(CSV_FILE, encoding='utf-8') # encoding='utf-8' ekliyoruz

if data.empty:
    print(f"{CSV_FILE} dosyası boş.")
else:
    X = data.drop('etiket', axis=1) 
    y = data['etiket']

    if y.nunique() <= 1 or y.groupby(y).size().min() < 2:
        print(f"HATA: Yeterli veri veya etiket yok. Her etiketten en az 2 adet toplayın.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = KNeighborsClassifier(n_neighbors=5)
    
    print("Model eğitiliyor (İki Elli)...")
    model.fit(X_train, y_train)
    print("Model eğitildi.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Modelin Test Doğruluğu: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_FILE)
    print(f"Model '{MODEL_FILE}' dosyasına kaydedildi.")