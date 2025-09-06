 
# Pusula Talent Academy Data Science Case Study
Emine Yiğit
emine.ygt@outlook.com

Bu proje, bir fiziksel tıp ve rehabilitasyon merkezine ait **2235 hasta gözlemini** içeren veri seti üzerinde kapsamlı bir **Keşifsel Veri Analizi (EDA)** ve **veri ön işleme** çalışmasını içermektedir.  

**Projenin temel amacı:** Veriyi anlamak, içgörüler çıkarmak ve potansiyel bir makine öğrenmesi modeli için veriyi temiz, tutarlı ve analize hazır hâle getirmektir.

**Hedef Değişken:** `TedaviSuresi`

Proje, Python'da `pandas`, `matplotlib`, `seaborn`, `scikit-learn` ve `ydata-profiling` gibi temel veri bilimi kütüphaneleri kullanılarak geliştirilmiştir. Tüm analiz ve ön işleme adımları, yeniden kullanılabilir fonksiyonlar halinde modüler bir yapıda tasarlanmış ve tek bir ana pipeline (`run_pipeline`) üzerinden çalıştırılmıştır.

---

## 1. Keşifsel Veri Analizi (EDA)

Bu aşamada veri setinin yapısını, dağılımlarını ve değişkenler arası ilişkileri anlamak için aşağıdaki analizler yapılmıştır:

- **Temel İstatistikler:** Veri setinin boyutu, sütunları ve veri tipleri incelendi.  
- **Eksik Değer Analizi:** Eksik verilerin tespiti, sayısal ve görsel olarak görselleştirilmesi.  
- **Hedef Değişken Analizi:** `TedaviSuresi`'nin dağılımı, aykırı değerleri ve istatistikleri incelendi.  
- **Sayısal ve Kategorik Değişken Analizi:** `Yaş`, `Cinsiyet`, `KanGrubu` gibi özelliklerin dağılımları görselleştirildi.  
- **Korelasyon Analizi:** Değişkenler arasındaki doğrusal ilişkiler bir ısı haritası ile incelendi.  
- **Detaylı Analizler:** Bölümlere göre tedavi süreleri, en sık görülen tanılar ve tedaviler analiz edildi.

Grafikler çıktıları klasöründe yer almaktadır.

### EDA HTML Raporu
EDA adımlarının ardından, `ydata-profiling` kullanılarak interaktif bir HTML raporu oluşturulmuştur. Bu rapor:

- Her bir değişkenin dağılımını ve istatistiklerini gösterir  
- Eksik veri durumlarını içerir  
- Değişkenler arası ilişkileri analiz eder  

Raporumuz EDA çıktıları klasöründe yer almaktadır.

---

## 2. Veri Ön İşleme

EDA'dan elde edilen bulgular doğrultusunda, veri setini modellemeye hazır hâle getirmek için aşağıdaki adımlar uygulanmıştır:

- **Eksik Değerlerin Doldurulması:** `KronikHastalik` ve `Alerji` gibi sütunlardaki boş değerler `"Yok"` gibi mantıksal bir değerle dolduruldu.  
- **Metin Verilerinin Standardizasyonu:** `Alerji` gibi sütunlardaki farklı yazımlar standart hâle getirildi (ör. `"toz"` ve `"TOZ"`).  
- **Özellik Mühendisliği (Feature Engineering):**  
  - `TedaviSuresi` ve `UygulamaSuresi` gibi metin tabanlı sütunlardan sayısal değerler (`TedaviSuresi_Sayi`) çıkarıldı.  
  - Bir hastanın sahip olduğu tanı sayısı (`Tanilar_Sayisi`) gibi yeni özellikler türetildi.  
- **Kategorik Değişkenlerin Kodlanması:**  
  - **Label Encoding:** `Cinsiyet` gibi ikili kategoriler 0 ve 1'e dönüştürüldü.  
  - **One-Hot Encoding:** `KanGrubu` gibi nominal kategoriler için kukla değişkenler oluşturuldu.  
- **Sayısal Özelliklerin Ölçeklendirilmesi:** Farklı ölçeklerdeki sayısal değişkenler `StandardScaler` kullanılarak standartlaştırıldı.  

---

##  Gerekli Kütüphaneler

Projeyi çalıştırmak için aşağıdaki Python kütüphaneleri gereklidir:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl ydata-profiling

##  Çıktılar

- **`processed_data.csv`**: Tüm ön işleme adımlarından geçmiş, modellemeye hazır temiz veri seti.

- **`eda_report.html`**: `ydata-profiling` tarafından oluşturulan, veri setindeki her bir değişken için detaylı istatistikler, dağılımlar ve etkileşimler içeren  rapor.

