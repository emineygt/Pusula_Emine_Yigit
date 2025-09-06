import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from ydata_profiling import ProfileReport

# Türkçe karakterler için matplotlib ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')


def load_data(file_path):
    print("Veri yükleniyor...")
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path, engine='openpyxl')
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        raise ValueError("Desteklenen dosya formatları: .xlsx, .xls, .csv")
    
    data.columns = data.columns.str.strip()
    print(f"✓ Veri yüklendi: {data.shape[0]} gözlem, {data.shape[1]} özellik")
    return data

def show_basic_info(data):
    print("=" * 60)
    print("TEMEL VERİ BİLGİLERİ")
    print("=" * 60)
    
    print(f"Boyut: {data.shape[0]} satır x {data.shape[1]} sütun\n")
    print("Sütunlar:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i:2d}. {col:<25} ({data[col].dtype})")
    print(f"\nİlk 5 kayıt:")
    print(data.head())
    return data

# =================== EDA ===================

# --- Eksik Değer Analizi ---
def analyze_missing_values(data):
    print("\n" + "="*50)
    print("EKSİK DEĞER ANALİZİ")
    print("="*50)
    
    missing_stats = data.isnull().sum()
    missing_percentage = (missing_stats / len(data) * 100).round(2)
    missing_df = pd.DataFrame({'Eksik_Sayı': missing_stats, 'Eksik_Yüzde': missing_percentage})
    
    print(missing_df[missing_df['Eksik_Sayı'] > 0])
    
    plt.figure(figsize=(12, 6))
    missing_data = missing_stats[missing_stats > 0].sort_values(ascending=False)
    if len(missing_data) > 0:
        plt.subplot(1, 2, 1)
        missing_data.plot(kind='bar')
        plt.title('Eksik Değer Sayıları')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        (missing_data / len(data) * 100).plot(kind='bar', color='orange')
        plt.title('Eksik Değer Yüzdeleri')
        plt.ylabel('Yüzde (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10,6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Eksik Değer Haritası")
        plt.show()
    else:
        plt.text(0.5, 0.5, 'Hiç eksik değer bulunamadı!', ha='center', va='center', fontsize=14)

# --- Tekrarlı Kayıt Analizi ---
def analyze_duplicates(data):
    print("\n" + "="*50)
    print("TEKRARLI KAYIT ANALİZİ")
    print("="*50)
    
    total_duplicates = data.duplicated().sum()
    print(f"Toplam tekrar eden kayıt sayısı: {total_duplicates}")
    
    if 'HastaNo' in data.columns and 'TedaviAdi' in data.columns:
        dup_hasta = data[data.duplicated(subset=['HastaNo','TedaviAdi'], keep=False)]
        print("Aynı Hasta + Aynı Tedavi tekrar eden kayıtlar:")
        print(dup_hasta)

# --- Hedef Değişken Analizi ---
def analyze_target_variable(data, target_col='TedaviSuresi'):
    print("\n" + "="*50)
    print("HEDEF DEĞİŞKEN ANALİZİ")
    print("="*50)
    
    if target_col not in data.columns:
        print(f"UYARI: '{target_col}' sütunu bulunamadı!")
        return
    
    target_counts = data[target_col].value_counts().sort_index()
    print("Tedavi Süresi Dağılımı:")
    print(target_counts)
    
    data_copy = data.copy()
    data_copy['TedaviSuresi_Sayi'] = data_copy[target_col].str.extract('(\d+)').astype(float)
    
    print(f"\nTedavi Süresi İstatistikleri:")
    print(f"Ortalama: {data_copy['TedaviSuresi_Sayi'].mean():.2f} seans")
    print(f"Medyan: {data_copy['TedaviSuresi_Sayi'].median():.2f} seans")
    print(f"Standart Sapma: {data_copy['TedaviSuresi_Sayi'].std():.2f} seans")
    print(f"Min-Max: {data_copy['TedaviSuresi_Sayi'].min()}-{data_copy['TedaviSuresi_Sayi'].max()} seans")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    target_counts.plot(kind='bar', color='skyblue')
    plt.title('Tedavi Süresi Dağılımı')
    plt.xlabel('Tedavi Süresi')
    plt.ylabel('Frekans')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    data_copy['TedaviSuresi_Sayi'].hist(bins=20, color='lightgreen', alpha=0.7)
    plt.title('Tedavi Süresi Histogramı')
    plt.xlabel('Seans Sayısı')
    plt.ylabel('Frekans')
    
    plt.subplot(1, 3, 3)
    data_copy.boxplot(column='TedaviSuresi_Sayi', ax=plt.gca())
    plt.title('Tedavi Süresi Box Plot')
    plt.ylabel('Seans Sayısı')
    
    plt.tight_layout()
    plt.show()

# --- Yaş Dağılımı Analizi ---
def analyze_age_distribution(data, age_col='Yas'):
    print("\n" + "="*50)
    print("YAŞ DAĞILIMI ANALİZİ")
    print("="*50)
    
    if age_col not in data.columns:
        print(f"UYARI: '{age_col}' sütunu bulunamadı!")
        return
    
    print(data[age_col].describe())
    
    Q1 = data[age_col].quantile(0.25)
    Q3 = data[age_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[age_col] < Q1 - 1.5*IQR) | (data[age_col] > Q3 + 1.5*IQR)]
    print(f"Aykırı değer sayısı: {len(outliers)}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1,3,1)
    data[age_col].hist(bins=30, color='coral', alpha=0.7)
    plt.title('Yaş Dağılımı')
    
    plt.subplot(1,3,2)
    data.boxplot(column=age_col, ax=plt.gca())
    plt.title('Yaş Box Plot')
    
    if 'Cinsiyet' in data.columns:
        plt.subplot(1,3,3)
        data.boxplot(column=age_col, by='Cinsiyet', ax=plt.gca())
        plt.title('Cinsiyete Göre Yaş Dağılımı')
        plt.suptitle('')
    
    plt.tight_layout()
    plt.show()

# --- Kategorik Değişkenler Analizi ---
def analyze_categorical_features(data):
    print("\n" + "="*50)
    print("KATEGORİK DEĞİŞKENLER ANALİZİ")
    print("="*50)
    
    categorical_cols = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum']
    available_cols = [col for col in categorical_cols if col in data.columns]
    
    if not available_cols:
        print("Kategorik sütunlar bulunamadı!")
        return
    
    plt.figure(figsize=(20, 15))
    
    for i, col in enumerate(available_cols, 1):
        value_counts = data[col].value_counts()
        print(f"\n{col} Dağılımı:")
        print(value_counts)
        print(f"Benzersiz değer sayısı: {data[col].nunique()}")
        
        plt.subplot(3, 2, i)
        if len(value_counts) <= 10:
            value_counts.plot(kind='bar', ax=plt.gca())
        else:
            value_counts.head(10).plot(kind='bar', ax=plt.gca())
        plt.xticks(rotation=45)
        plt.title(f'{col} Dağılımı')
        plt.ylabel('Frekans')
    
    plt.tight_layout()
    plt.show()

# --- Korelasyon Analizi ---
def analyze_correlations(data):
    print("\n" + "="*50)
    print("KORELASYON ANALİZİ")
    print("="*50)
    
    data_encoded = data.copy()
    le = LabelEncoder()
    categorical_cols = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'TedaviSuresi']
    
    for col in categorical_cols:
        if col in data_encoded.columns:
            data_encoded[f'{col}_encoded'] = le.fit_transform(data_encoded[col].astype(str))
    
    numeric_cols = ['Yas'] + [col for col in data_encoded.columns if col.endswith('_encoded')]
    corr_matrix = data_encoded[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.3f')
    plt.title('Değişkenler Arası Korelasyon Matrisi')
    plt.tight_layout()
    plt.show()
    
    if 'TedaviSuresi_encoded' in corr_matrix.columns:
        target_corr = corr_matrix['TedaviSuresi_encoded'].abs().sort_values(ascending=False)
        print("Tedavi Süresi ile En Yüksek Korelasyonlar:")
        print(target_corr[target_corr.index != 'TedaviSuresi_encoded'])

# --- Tedavi Süresi Detaylı Analizi ---
def analyze_treatment_duration(data, col='TedaviSuresi'):
    print("\n" + "="*50)
    print("TEDAVİ SÜRESİ ANALİZİ")
    print("="*50)
    
    if col not in data.columns:
        print(f"UYARI: '{col}' sütunu bulunamadı!")
        return
    
    data_copy = data.copy()
    data_copy['TedaviSuresiNum'] = data_copy[col].str.extract('(\d+)').astype(float)
    
    plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    sns.histplot(data_copy['TedaviSuresiNum'], bins=15, kde=True)
    plt.title('Tedavi Süresi Dağılımı')
    
    plt.subplot(1,2,2)
    if 'Bolum' in data_copy.columns:
        sns.boxplot(x='Bolum', y='TedaviSuresiNum', data=data_copy, palette="Set2")
        plt.xticks(rotation=45)
        plt.title('Bölümlere Göre Tedavi Süresi')
    
    plt.tight_layout()
    plt.show()

# --- Tanılar ve Tedaviler Görselleştirme ---
def analyze_top_treatments_and_diagnoses(data, treatment_col='TedaviAdi', diagnosis_col='Tanilar'):
    print("\n" + "="*50)
    print("TANI VE TEDAVİ ANALİZİ")
    print("="*50)
    
    if treatment_col in data.columns:
        top_treatments = data[treatment_col].value_counts().head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(y=top_treatments.index, x=top_treatments.values, palette="mako", legend=False)
        plt.title("En Çok Uygulanan 10 Tedavi")
        plt.xlabel("Hasta Sayısı")
        plt.ylabel("Tedavi Adı")
        plt.tight_layout()
        plt.show()
    
    if diagnosis_col in data.columns:
        top_diagnoses = data[diagnosis_col].value_counts().head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(y=top_diagnoses.index, x=top_diagnoses.values, palette="crest", legend=False)
        plt.title("En Sık Görülen 10 Tanı")
        plt.xlabel("Hasta Sayısı")
        plt.ylabel("Tanı")
        plt.tight_layout()
        plt.show()

# =================== VERİ ÖN İŞLEME ===================

def clean_missing_values(data):
    print("\n" + "="*50)
    print("EKSİK DEĞER TEMİZLEME")
    print("="*50)
    
    data_clean = data.replace('', np.nan)
    
    for col, default in [('KronikHastalik', 'Yok'), ('Alerji', 'Yok'), ('UygulamaYerleri', 'Belirtilmemiş')]:
        if col in data_clean.columns:
            if col == 'Alerji':
                # Büyük harfe çevir ve boş/nan olanları 'Yok' ile doldur
                data_clean[col] = data_clean[col].astype(str).str.upper().replace('NAN', 'Yok')
            else:
                data_clean[col] = data_clean[col].fillna(default)
    
    print(f"Temizlik sonrası eksik değer: {data_clean.isnull().sum().sum()}")
    return data_clean

def extract_numerical_features(data):
    print("\n" + "="*50)
    print("SAYISAL ÖZELLİK ÇIKARIMI")
    print("="*50)
    
    data_with_nums = data.copy()
    
    if 'TedaviSuresi' in data.columns:
        data_with_nums['TedaviSuresi_Sayi'] = data_with_nums['TedaviSuresi'].str.extract('(\d+)').astype(float)
    if 'UygulamaSuresi' in data.columns:
        data_with_nums['UygulamaSuresi_Sayi'] = data_with_nums['UygulamaSuresi'].str.extract('(\d+)').astype(float)
    if 'KronikHastalik' in data.columns:
        data_with_nums['KronikHastalik_Sayisi'] = data_with_nums['KronikHastalik'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    if 'Tanilar' in data.columns:
        data_with_nums['Tanilar_Sayisi'] = data_with_nums['Tanilar'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    
    if 'Yas' in data.columns and 'TedaviSuresi_Sayi' in data_with_nums.columns:
        data_with_nums['Yas_TedaviSuresi_Ratio'] = data_with_nums['Yas'] / (data_with_nums['TedaviSuresi_Sayi'] + 1e-6)
    
    return data_with_nums

def encode_categorical_features(data):
    print("\n" + "="*50)
    print("KATEGORİK ÖZELLİK ENCODING")
    print("="*50)
    
    data_encoded = data.copy()
    le = LabelEncoder()
    
    for col in ['Cinsiyet', 'Uyruk']:
        if col in data_encoded.columns:
            data_encoded[f'{col}_LE'] = le.fit_transform(data_encoded[col].astype(str))
    
    if 'KanGrubu' in data_encoded.columns:
        data_encoded = pd.concat([data_encoded, pd.get_dummies(data_encoded['KanGrubu'], prefix='KanGrubu')], axis=1)
    
    for col in ['Alerji','KronikHastalik']:
        if col in data_encoded.columns:
            data_encoded[f'{col}_Var'] = data_encoded[col].apply(lambda x: 0 if str(x).lower() in ['yok','none','nan'] else 1)
    
    return data_encoded

def scale_numerical_features(data, cols_to_scale=None):
    print("\n" + "="*50)
    print("SAYISAL ÖZELLİK STANDARTLAŞTIRMA")
    print("="*50)
    
    data_scaled = data.copy()
    scaler = StandardScaler()
    
    if not cols_to_scale:
        cols_to_scale = [col for col in data_scaled.select_dtypes(include=np.number).columns if col not in ['HastaNo']]
    
    data_scaled[cols_to_scale] = scaler.fit_transform(data_scaled[cols_to_scale])
    
    return data_scaled



def run_pipeline(file_path):
    data = load_data(file_path)
    show_basic_info(data)
    
    analyze_missing_values(data)
    analyze_duplicates(data)
    analyze_target_variable(data)
    analyze_age_distribution(data)
    analyze_categorical_features(data)
    analyze_correlations(data)
    analyze_treatment_duration(data)
    analyze_top_treatments_and_diagnoses(data)

    # HTML EDA raporu oluştur
    print("\n✓ HTML EDA raporu oluşturuluyor...")
    profile = ProfileReport(data, title="Otomatik EDA Raporu", explorative=True)
    profile.to_file("eda_report.html")
    print("✓ HTML rapor kaydedildi: eda_report.html")
    
    data_clean = clean_missing_values(data)
    data_nums = extract_numerical_features(data_clean)
    data_encoded = encode_categorical_features(data_nums)
    data_scaled = scale_numerical_features(data_encoded)
    
    output_file = 'processed_data.csv'
    data_scaled.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=1)
    print(f"\n✓ İşlenmiş veri '{output_file}' olarak kaydedildi.")
    
    return data_scaled


processed_data = run_pipeline('Talent_Academy_Case_DT_2025.xlsx')
