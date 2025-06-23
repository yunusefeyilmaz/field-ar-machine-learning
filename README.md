# Su Stresi Tahmin Projesi

Bu proje, uydu görüntüleri ve hava durumu verilerine dayanarak tarım alanlarındaki su stresi seviyelerini tahmin etmeyi amaçlamaktadır. Sistem, geçmiş uydu su stresi görüntülerini analiz eder, bu verileri hava durumu bilgileriyle birleştirir ve gelecekteki su stresi seviyelerini tahmin etmek için bir makine öğrenimi modeli eğitir.

## Proje Yapısı

```
water_stress_prediction/
├── data/
│   ├── satellite/          - Uydu görüntüleri (format: YYYY-MM-DD_WaterStress.png)
│   ├── weather/            - Hava durumu verileri (data.json)
│   └── prepared/           - Hazırlanmış veri setleri (otomatik oluşturulur)
├── models/                 - Eğitilmiş ML modelleri (otomatik oluşturulur)
├── notebooks/              - Keşif amaçlı Jupyter defterleri
├── output/                 - Çıktı dosyaları ve tahmin sonuçları
└── src/                    - Kaynak kodlar
    ├── build_dataset.py    - Ham veriden veri seti oluşturma betiği paralel
    ├── build_dataset.py    - Ham veriden veri seti oluşturma betiği
    ├── normalize           - Veri normalizasyonu
    ├── config.py           - Konfigürasyon ayarları
    ├── predict_water.py    - Tahmin yapma betiği basitleştirilmiş
    ├── predict.py          - Tahmin yapma betiği
    ├── run_pipeline.py     - Tüm hattı çalıştıran ana betik
    ├── server.py           - Oluşturulan modele api üzerinde erişim için
    ├── visulation.py       - Veri seti görselleştirme
    └── train_model.py      - ML modelini eğiten betik

```

## Veri Açıklaması

### Uydu Verisi

- Format: Su stresi seviyelerini renk kodlarıyla gösteren PNG görüntüler
- Renk kodları:
  - RGB(228, 0, 2): TOPRAK (değer yok)
  - RGB(255, 86, 0): YÜKSEK STRES (1.0)
  - RGB(107, 254, 147): ORTA STRES (0.5)
  - RGB(0, 239, 254): DÜŞÜK STRES (0.2)
  - RGB(0, 0, 143): STRES YOK (0.0)

### Hava Durumu Verisi

- Format: Günlük hava parametrelerini içeren JSON dosyası
- Temel parametreler:
  - temperature_2m_max, temperature_2m_mean, temperature_2m_min
  - precipitation_sum, precipitation_hours
  - soil_moisture_0_to_10cm_mean
  - relative_humidity değerleri
  - sunshine_duration
  - wind_speed_10m_max
  - et0_fao_evapotranspiration_sum
  - ve daha fazlası

## Kurulum

1. Depoyu klonlayın:

   ```
   git clone https://github.com/yunusefeyilmaz/field-ar-machine-learning.git
   cd field-ar-machine-learning
   ```

2. Bağımlılıkları yükleyin:
   ```
   pip install -r requirements.txt
   ```

## Kullanım

### Tüm Hattı Çalıştırmak

Tüm hattı (veri seti oluşturma, model eğitimi ve tahmin) çalıştırmak için:

```
python src/run_pipeline.py
```

### Sadece Veri Seti Oluşturmak

```
python src/run_pipeline.py --build-dataset
```

### Sadece Modeli Eğitmek

```
python src/run_pipeline.py --train-model --days 7
```

### Sadece Tahmin Yapmak

```
python src/run_pipeline.py --predict --bbox 28.9,40.15,29.1,40.25 --days 7
```

Opsiyonel parametreler:

- `--current-stress`: Biliniyorsa mevcut su stresi değeri (0.0-1.0)
- `--days`: Kaç gün sonrası için tahmin yapılacak (varsayılan: 7)
- `--bbox`: Sınır kutusu koordinatları: `lon_min,lat_min,lon_max,lat_max` formatında

## Nasıl Çalışır?

1. **Veri Seti Oluşturma**:

   - Uydu görüntüleri işlenerek su stresi seviyeleri çıkarılır
   - Görüntüler yaklaşık 100x100 piksellik ızgara hücrelerine bölünür
   - Her hücre için, piksel renklerine göre ortalama su stresi seviyesi hesaplanır
   - Bu veriler, aynı tarihlerdeki hava durumu bilgileriyle birleştirilir
   - Model eğitimi için özellikler içeren bir CSV veri seti oluşturulur

2. **Model Eğitimi**:

   - Makine öğrenimi modeli (Random Forest Regressor) bu veri setiyle eğitilir
   - Model, mevcut verilere göre X gün sonraki su stresi seviyesini tahmin etmeyi öğrenir
   - Su stresini etkileyen temel faktörleri anlamak için özellik önemi analiz edilir

3. **Tahmin**:
   - Eğitilmiş model, bir sınır kutusu (coğrafi koordinatlar) alır
   - Tahmin için hava durumu tahmin verileri kullanılır
   - Model, X gün sonraki su stresi seviyesini tahmin eder

## Çıktı Dosyaları

Hat çalıştırıldığında aşağıdaki çıktılar üretilir:

- `water_stress_dataset.csv`: Eğitim için tam veri seti
- `water_stress_prediction_7days.joblib`: Eğitilmiş model (varsayılan 7 gün tahmini)
- `feature_importance_7days.png`: Özellik önemini gösteren grafik
- `actual_vs_predicted_7days.png`: Gerçek ve tahmin edilen değerlerin karşılaştırması
- `model_info_7days.json`: Model performans metrikleri ve parametreler
- `prediction_result.json`: Son tahmin sonuçları

## Gereksinimler

Tüm bağımlılıkların listesi için `requirements.txt` dosyasına bakınız.

## Sunucu

Sunucuyu çalıştırmak için aşağıdaki komutu kullanın:

```
uvicorn src.server:app --reload
```

Bu komut, geliştirme için otomatik yeniden yükleme ile FastAPI sunucusunu başlatır.
