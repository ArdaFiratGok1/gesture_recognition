# GestureApp: Gerçek Zamanlı El ve Yüz Hareketi Tanıma Sistemi

## Proje Tanımı

Bu proje, OpenCV, MediaPipe ve Scikit-learn kütüphanelerini kullanarak geliştirilmiş, gerçek zamanlı bir jest ve poz tanıma uygulamasıdır. Sistem, bir web kamerası aracılığıyla kullanıcının el (iki el) ve yüz kilit noktalarını analiz eder. Bu veriyi, önceden eğitilmiş bir makine öğrenimi modeline (`.pkl` dosyası) gönderir ve tanınan poza karşılık gelen, önceden yapılandırılmış bir görsel (ve/veya ses) çıktısı üretir.

Uygulama, hem son kullanıcılar için doğrudan çalıştırılabilir bir `.exe` dosyası hem de geliştiriciler için yeniden eğitilebilir bir Python kaynak kodu olarak yapılandırılmıştır.

## İçindekiler

1.  **Son Kullanıcı Yönergeleri (EXE)**
    * Sistem Gereksinimleri
    * Gerekli Dosya Yapısı
    * Uygulamanın Çalıştırılması
    * Sorun Giderme
2.  **Geliştirici Yönergeleri (Kaynak Kod)**
    * Teknik Gereksinimler
    * Kurulum ve Bağımlılıklar
    * Mevcut Model ile Çalıştırma
3.  **Modeli Yeniden Eğitme Kılavuzu**
    * Adım 1: Veri Toplama (`1_veri_topla.py`)
    * Adım 2: Modeli Eğitme (`2_model_egit.py`)
    * Adım 3: Aksiyonları Yapılandırma (`3_tahmin_et.py`)
4.  **Uygulamayı `.exe` Olarak Paketleme**
    * PyInstaller Kurulumu
    * Build Komutu
5.  **Kullanılan Teknolojiler**
6.  **Proje Sahibi**

---

## 1. Son Kullanıcı Yönergeleri (EXE)

Bu bölüm, uygulamayı Python veya herhangi bir kütüphane kurmadan, doğrudan çalıştırmak isteyen kullanıcılar içindir.

### Sistem Gereksinimleri

* Windows 10 veya 11 işletim sistemi.
* Sisteme bağlı ve çalışan bir web kamerası.

### Gerekli Dosya Yapısı

Uygulamanın (`GestureApp.exe`) hatasız çalışabilmesi için, bulunduğu klasör (`dist/GestureApp` olarak varsayılmıştır) aşağıdaki dosya ve klasörleri **birebir** içermelidir:

```
📁 dist/GestureApp/
│
├── 📂 images/
│   ├── neutral_monke.png
│   ├── thinking_monke.png
│   ├── finding_monke.jpeg
│   ├── scared_monke.png
│   └── ... (diğer tüm resimler)
│
├── 📂 sounds/
│   └── ... (eğer kullanılıyorsa tüm ses dosyaları)
│
├── 🚀 GestureApp.exe               (Ana Uygulama)
│
├── 🧠 iki_elli_model.pkl         (Eğitilmiş Makine Öğrenimi Modeli)
│
└── ... (Uygulamanın çalışması için gerekli diğer .dll ve kütüphane dosyaları)
```

**Özetle:** `.exe` dosyası, `iki_elli_model.pkl` dosyası ve `images` (ve kullanılıyorsa `sounds`) klasörü ile **aynı dizin seviyesinde** olmalıdır.

### Uygulamanın Çalıştırılması

1.  Gerekli tüm dosyaları içeren `dist/GestureApp` klasörünü açın.
2.  `GestureApp.exe` dosyasına çift tıklayın.
3.  Uygulama başladığında iki pencere açılacaktır:
    * **"Gercek Zamanli Tahmin"**: Sizin kamera görüntünüzü ve sistemin o anki tahminini (`Tahmin: ...`) gösteren ana pencere.
    * **"Aksiyon"**: Tanınan harekete karşılık gelen görüntüyü gösteren ikinci pencere.

### Sorun Giderme

Eğer `.exe` dosyası çalışmazsa, aşağıdaki hata mesajlarını alabilirsiniz:

* **"Kritik Hata: Model dosyası bulunamadı!"**:
    * **Sebep:** `iki_elli_model.pkl` dosyası, `.exe` ile aynı klasörde değil.
    * **Çözüm:** Yukarıdaki "Gerekli Dosya Yapısı" bölümünü kontrol edin ve `.pkl` dosyasını `.exe`'nin yanına kopyalayın.

* **"Kritik Hata: Kamera (0) açılamadı"**:
    * **Sebep:** Kamera bağlı değil veya başka bir uygulama (Örn: Zoom, Discord, Teams) tarafından aktif olarak kullanılıyor.
    * **Çözüm:** Kamerayı kullanan diğer tüm uygulamaları kapatın ve `GestureApp.exe`'yi yeniden başlatın.

* **Aksiyon Penceresi Siyah Gösteriliyor / Resimler Görünmüyor**:
    * **Sebep:** `images` klasörü `.exe` ile aynı klasörde değil veya `tahmin_et.py` içindeki `AKSIYONLAR` sözlüğünde belirtilen dosya adları (`neutral_monke.png` vb.) `images` klasöründeki dosya adlarıyla eşleşmiyor.
    * **Çözüm:** Dosya yapısını kontrol edin.

---

## 2. Geliştirici Yönergeleri (Kaynak Kod)

Bu bölüm, projeyi kaynak kodundan çalıştırmak, değiştirmek veya yeniden eğitmek isteyen geliştiriciler içindir.

### Teknik Gereksinimler

* **Python 3.11** (Proje bu sürümle geliştirilmiş ve test edilmiştir)
* `pip` (Python Paket Yöneticisi)
* Bir web kamerası

### Kurulum ve Bağımlılıklar

1.  **Projeyi Klonlayın veya İndirin:**
    ```bash
    git clone [Projenizin GitHub Adresi]
    cd gesture_ml
    ```

2.  **Sanal Ortam (Virtual Environment) Oluşturun:**
    Proje bağımlılıklarını sisteminizden izole etmek ve `mediapipe`'in olası dosya yolu hatalarını önlemek için sanal ortam kullanılması *şiddetle* tavsiye edilir.
    ```bash
    # 'venv' adında bir sanal ortam oluştur
    python -m venv venv
    ```

3.  **Sanal Ortamı Aktif Edin (Windows PowerShell):**
    ```bash
    .\venv\Scripts\activate
    ```
    (Terminal satırınızın başında `(venv)` ibaresi görünmelidir.)

4.  **Gerekli Kütüphaneleri Kurun:**
    ```bash
    (venv) PS C:\gesture_ml> pip install opencv-python mediapipe scikit-learn pandas pygame
    ```

### Mevcut Model ile Çalıştırma

Modeli yeniden eğitmeden, doğrudan `tahmin_et.py` betiğini çalıştırmak için:
```bash
(venv) PS C:\gesture_ml> python tahmin_et.py
```
Bu komut, `iki_elli_model.pkl` dosyasını yükleyecek ve kamerayı açacaktır.

---

## 3. Modeli Yeniden Eğitme Kılavuzu

Modelin tanımasını istediğiniz yeni pozları (jestleri) eklemek veya mevcutları iyileştirmek için bu üç adımlı süreci izleyin.

### Adım 1: Veri Toplama (`1_veri_topla.py`)

Bu betik, kamerayı açar ve klavye tuşlarına basarak pozlarınızı etiketlemenizi sağlar.

1.  **`1_veri_topla.py`** dosyasını bir kod editörü ile açın.
2.  `ETIKETLER` sözlüğünü (dictionary) bulun.
3.  Eklemek istediğiniz yeni pozları, benzersiz bir klavye tuşu atayarak buraya ekleyin:
    ```python
    ETIKETLER = {
        '0': 'neutral',
        '1': 'thinking',
        '2': 'finding',
        '3': 'scared',
        '4': 'alkis'  # <-- YENİ EKLENEN POZ
    }
    ```
4.  **Çok Önemli:** Eğer sıfırdan bir eğitim yapıyorsanız, `C:\gesture_ml\` klasöründeki mevcut `iki_elli_pozlar.csv` dosyasını **silin**.
5.  Terminalde `1_veri_topla.py` betiğini çalıştırın:
    ```bash
    (venv) PS C:\gesture_ml> python 1_veri_topla.py
    ```
6.  Program size hangi tuşun hangi pozu kaydettiğini gösterecektir. Kameraya bakın, öğretmek istediğiniz pozu yapın ve ilgili tuşa art arda basın.

**Veri Toplama İçin En İyi Pratikler:**
* **Çeşitlilik:** Bir poz için 100 veri topluyorsanız, 100'ü de birebir aynı olmasın. Elinizi/yüzünüzü hafifçe oynatın, farklı açılardan (hafif sağa/sola) pozu verin.
* **"Neutral" (Boş) Sınıfı:** `neutral` etiketi, modelin en çok ihtiyaç duyduğu veridir. Bu etiket için veri toplarken normal durun, elleriniz kamerada görünmesin, anlamsız hareketler yapın. Bu, modelin "tanımlı olmayan" duruşları ayırt etmesini sağlar.
* **Minimum Veri:** Her etiket için en az 100-200 adet çeşitli veri örneği toplanması tavsiye edilir.

### Adım 2: Modeli Eğitme (`2_model_egit.py`)

Bu betik, topladığınız CSV dosyasını okur ve modeli eğitir.

1.  Veri toplamayı bitirdikten sonra ('q' ile çıkın), `2_model_egit.py` betiğini çalıştırın:
    ```bash
    (venv) PS C:\gesture_ml> python 2_model_egit.py
    ```
2.  Bu script, `iki_elli_pozlar.csv` dosyasını okuyacak ve `iki_elli_model.pkl` adında yeni bir model dosyası oluşturacaktır. Terminalde modelin test doğruluğunu göreceksiniz.
3.  *(İsteğe bağlı)* Daha yüksek doğruluk için `2_model_egit.py` içindeki `KNeighborsClassifier` yerine `RandomForestClassifier` modelini yorum satırından çıkarıp kullanabilirsiniz.

### Adım 3: Aksiyonları Yapılandırma (`tahmin_et.py`)

Modeliniz artık yeni pozunuzu (`alkis` gibi) tanıyabiliyor. Şimdi bu poza bir aksiyon (resim/ses) atamanız gerekiyor.

1.  **`tahmin_et.py`** dosyasını açın.
2.  `AKSIYONLAR` sözlüğünü bulun.
3.  `1_veri_topla.py`'de eklediğiniz yeni etiketi (`alkis`) buraya da ekleyin ve tetikleyeceği resim/ses dosyasının yolunu belirtin:
    ```python
    AKSIYONLAR = {
        'neutral': { ... },
        'thinking': { ... },
        # ...
        'alkis': {  # <-- YENİ EKLENEN AKSİYON
            'ses': os.path.join(base_path, 'sounds', 'alkis.wav'),
            'görüntü': os.path.join(base_path, 'images', 'alkis_resmi.png')
        }
    }
    ```
4.  Yeni resim ve ses dosyalarınızı `images` ve (eğer oluşturduysanız) `sounds` klasörlerine ekleyin.
5.  Programı çalıştırarak yeni pozunuzun tanınıp tanınmadığını test edin.

---

## 4. Uygulamayı `.exe` Olarak Paketleme

Geliştirmeyi tamamladıktan ve modelinizi eğittikten sonra, projeyi dağıtmak için `PyInstaller` kullanabilirsiniz.

1.  **PyInstaller Kurulumu:**
    ```bash
    (venv) PS C:\gesture_ml> pip install pyinstaller
    ```

2.  **`mediapipe` Modül Yolunu Bulun:**
    PyInstaller'ın, `mediapipe`'in gizli model dosyalarını bulması için ona tam yolu vermelisiniz. Bu yolu almak için aşağıdaki komutu çalıştırın:
    ```bash
    (venv) PS C:\gesture_ml> python -c "import mediapipe as mp; import os; print(os.path.join(os.path.dirname(mp.__file__), 'modules'))"
    ```
    (Çıktı şuna benzer olacaktır: `C:\gesture_ml\venv\Lib\site-packages\mediapipe\modules`)

3.  **Build Komutunu Çalıştırın:**
    Aşağıdaki komutu terminalde çalıştırın. `--add-data` içindeki `mediapipe` yolunu, bir üst adımda aldığınız çıktıyla güncelleyin.
    ```bash
    (venv) PS C:\gesture_ml> pyinstaller --name GestureApp --windowed --add-data "iki_elli_model.pkl;." --add-data "images;images" --add-data "sounds;sounds" --add-data "C:\gesture_ml\venv\Lib\site-packages\mediapipe\modules;mediapipe/modules" tahmin_et.py
    ```
    *(Not: Eğer `sounds` klasörü kullanmıyorsanız `--add-data "sounds;sounds"` kısmını silebilirsiniz.)*

4.  **Sonuç:**
    İşlem bittiğinde, çalıştırılabilir uygulamanız `dist\GestureApp` klasörünün içinde hazır olacaktır. Dağıtım yapmadan önce "Gerekli Dosya Yapısı" bölümündeki (Bölüm 1) gibi tüm dosyaların `.exe`'nin yanına doğru kopyalandığını manuel olarak kontrol edin.

---

## 5. Kullanılan Teknolojiler

* **Python 3.11:** Ana programlama dili.
* **OpenCV (opencv-python):** Kamera yönetimi, görüntü işleme ve görüntü gösterimi.
* **MediaPipe:** El ve yüz kilit noktalarının (landmarks) gerçek zamanlı tespiti.
* **Scikit-learn (sklearn):** Makine öğrenimi modeli (k-NN) oluşturma ve tahmin.
* **Pandas:** `.csv` veri setlerini okuma ve yönetme.
* **Pygame:** Ses dosyalarını (`.wav`, `.mp3`) çalma.
* **PyInstaller:** Python betiklerini Windows çalıştırılabilir dosyasına (`.exe`) paketleme.

---

## 6. Proje Sahibi

* **Arda Fırat Gök**
    * [GitHub Adresiniz Buraya]
    * [LinkedIn Adresiniz Buraya]
