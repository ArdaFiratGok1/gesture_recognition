# GestureApp: Gerçek Zamanlı Hareket Tanıma

## Kod üzerinden çalıştırma:

Bu bölüm, projeyi kaynak kodundan çalıştırmak, değiştirmek veya kendi pozlarınızı eğitmek istiyorsanız gereklidir.

**A. Kurulum (Sadece İlk Seferde)**

1.  **Sanal Ortam (Venv) Kur:**
    Kütüphanelerin sisteminize karışmaması için proje klasöründe (örn: `C:\gesture_ml`) bir sanal ortam oluşturun.

    ```bash
    # Terminali proje klasöründe açın
    python -m venv venv
    
    # Sanal ortamı aktif edin (Windows PowerShell için)
    .\venv\Scripts\activate
    ```
    Terminal satırınızın başında `(venv)` görmelisiniz.

2.  **Kütüphaneleri Yükle:**
    `venv` aktifken, gerekli tüm paketleri kurun:
    ```bash
    (venv) PS C:\gesture_ml> pip install opencv-python mediapipe scikit-learn pandas pygame
    ```

**B. Modeli Test Etme (Mevcut Modeli Çalıştırma)**

Eğer sadece mevcut modeli test edecekseniz, `venv` aktifken bu komutu çalıştırmanız yeterlidir:
```bash
(venv) PS C:\gesture_ml> python tahmin_et.py
```

**C. Kendi Modelini Eğitme (3 Adımda)**

Kendi pozlarınızı eklemek için bu sırayı takip edin:

**1. Veri Topla (`1_veri_topla.py`)**
* **Ne yapar:** Kamerayı açar ve bastığınız tuşlara göre el/yüz koordinatlarınızı `iki_elli_pozlar.csv` dosyasına kaydeder.
* **Nasıl:**
    1.  Önce `1_veri_topla.py` dosyasını açıp `ETIKETLER` sözlüğüne kendi pozlarınızı ekleyin (örn: `'5': 'selam_ver'`).
    2.  *(Önemli)* Yeni bir eğitim için eski `iki_elli_pozlar.csv` dosyasını silin.
    3.  `python 1_veri_topla.py` komutunu çalıştırın.
    4.  Kamerada pozu yaparken atadığınız tuşa (`'5'`) defalarca basın.
    5.  **Mutlaka 'neutral' (Boş duruş) pozu için de ('0' tuşu) bolca veri toplayın.** Bu, modelin kararlılığı için kritiktir.
    6.  'q' ile çıkın.

**2. Modeli Eğit (`2_model_egit.py`)**
* **Ne yapar:** `iki_elli_pozlar.csv` dosyasını okur ve `iki_elli_model.pkl` adında yeni bir model dosyası oluşturur.
* **Nasıl:** `1. adımı` bitirdikten hemen sonra bu komutu çalıştırın:
    ```bash
    (venv) PS C:\gesture_ml> python 2_model_egit.py
    ```
    Yeni `.pkl` dosyanız oluşacaktır.

**3. Aksiyonları Ayarla (`tahmin_et.py`)**
* **Ne yapar:** Modelin tanıdığı pozlara resim/ses atar.
* **Nasıl:**
    1.  `tahmin_et.py` dosyasını açıp `AKSIYONLAR` sözlüğüne gidin.
    2.  Yeni pozunuzu (`'selam_ver'` gibi) ve karşılığında göstereceği resim/ses dosyasının yolunu ekleyin.
    3.  `python tahmin_et.py` ile çalıştırıp test edin.
