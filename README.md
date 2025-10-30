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

---

## exe dosyasında çalıştırma:

Bu bölüm, Python veya kütüphaneleri kurmadan, uygulamayı doğrudan çalıştırmak isteyenler içindir.

**Kritik Kural: Dosya Konumları**
`GestureApp.exe`'nin çalışması için, tüm yardımcı dosyaların (`.pkl` modeli, `images` klasörü vb.) `.exe` ile **aynı klasörde** olması **gereklidir**.

**Klasör Yapınız Şöyle Görünmelidir:**
```
📁 GestureApp/  <-- (dist/GestureApp klasörünüz)
│
├──  GestureApp.exe        (Ana Uygulama)
│
├──  iki_elli_model.pkl  (Makine Öğrenimi Modeli)
│
├──  images/               (Resimlerin olduğu klasör)
│   ├── neutral_monke.png
│   ├── thinking_monke.png
│   └── ... (diğer tüm resimler)
│
├──  sounds/               (Eğer kullanıyorsanız ses klasörü)
│   └── ...
│
└── ... (Uygulamanın çalışması için gerekli diğer .dll dosyaları)
```

**Çalıştırma:**
Yukarıdaki dosya yapısı tamsa, `GestureApp.exe`'ye çift tıklamanız yeterlidir. Kamera ve aksiyon penceresi açılacaktır.

**Sık Karşılaşılan Hatalar:**

* **"Kritik Hata: Model dosyası bulunamadı!"**
    * **Çözüm:** `iki_elli_model.pkl` dosyası `.exe`'nin yanında değil. Kopyalayıp `.exe`'nin yanına yapıştırın.

* **"Kritik Hata: Kamera (0) açılamadı!"**
    * **Çözüm:** Kameranız başka bir program (Zoom, Discord vb.) tarafından kullanılıyor. O programları kapatıp `.exe`'yi yeniden başlatın.

* **Resimler görünmüyor (Aksiyon penceresi siyah)**
    * **Çözüm:** `images` klasörü `.exe`'nin yanında değil demektir. Klasörü kopyalayıp `.exe`'nin yanına yapıştırın.
