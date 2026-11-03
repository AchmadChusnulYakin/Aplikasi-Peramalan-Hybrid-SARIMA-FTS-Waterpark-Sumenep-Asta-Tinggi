# Aplikasi Peramalan Hybrid SARIMA–FTS · Waterpark Sumenep Asta Tinggi

Aplikasi analitik berbasis **Streamlit** untuk melakukan preprocessing, forecasting, evaluasi, diagnosis error, dan proyeksi masa depan jumlah kunjungan **Waterpark Sumenep (Asta Tinggi)** menggunakan **Hybrid SARIMA-FTS**.

Project ini merupakan implementasi forecasting dalam kegiatan **MBKM Skema Penelitian/Riset**. Aplikasi ini dirancang untuk peramalan dan analisis: pengguna dapat menggunakan dataset bawaan atau mengunggah dataset sendiri, memilih pembagian data, menjalankan pipeline Hybrid SARIMA-FTS secara kronologis, memeriksa hasil testing, melakukan diagnosis error, dan menyiapkan keluaran penelitian.

> **Catatan metodologis:** lapisan UI/UX berada di atas pipeline preprocessing, SARIMA, FTS, calibration, tuning, ensemble, evaluasi, dan future forecasting. Penyempurnaan tampilan tidak mengganti formula Hybrid SARIMA-FTS.

## 📌 Fitur Utama

### Dataset & Preprocessing
- Upload dataset **CSV, XLSX, atau XLS**.
- Auto-detection kolom periode/tanggal dan jumlah kunjungan.
- Mendukung format dua kolom (**tanggal/periode + jumlah kunjungan**) maupun format tiga kolom seperti Asta Tinggi (**No + Periode + Kunjungan**).
- Parsing tanggal ISO dan nama bulan dalam bahasa Indonesia.
- Nilai **0 dan Missing Value** diproses dengan **Interpolasi Linear** pada posisi yang dapat diinterpolasi.
- Tahun yang tidak tersedia pada sumber tidak dibuat secara sintetis.
- Audit preprocessing tersedia untuk meninjau perubahan data.
- Pembagian data tersedia secara kronologis: **70:30** dan **80:20**.

### Hybrid SARIMA-FTS
- **SARIMA:** konfigurasi awal `(2,1,0)(1,1,0,12)`, tetapi parameter tidak lagi dikunci di source code.
- **Parameter SARIMA dinamis:** pengguna dapat mengatur `p`, `d`, `q`, `P`, `D`, `Q`, dan `s` dari menu **02 Input Dataset** sebelum menjalankan pipeline.
- Rentang kontrol: `p` 0-5, `d` 0-2, `q` 0-5, `P` 0-3, `D` 0-2, `Q` 0-3, dan `s` 2-24.
- Komponen musiman dapat diaktifkan atau dinonaktifkan. Jika nonaktif, order musiman efektif menjadi `(0,0,0,0)`.
- Untuk riwayat yang belum memiliki sekurang-kurangnya dua siklus musiman dan 24 observasi, komponen musiman sementara tidak diestimasi pada origin tersebut; order non-musiman pilihan pengguna tetap digunakan.
- **FTS:** Chen Weighted.
- **Window:** `6 → 1` dan `12 → 1`.
- **Validation:** 12 observasi terakhir dari training.
- `n_intervals` FTS ditelusuri pada rentang **5-12**.
- `alpha` ditelusuri pada rentang **0,80-0,99** dengan interval **0,01**.
- Prediksi FTS dikalibrasi menggunakan **Linear Regression**.
- Konfigurasi dipilih pada validation dengan urutan kriteria **MAPE, MAE, RMSE**.
- Testing menggunakan **walk-forward/one-step-ahead** setelah konfigurasi ditetapkan.
- Forecast Hybrid menggunakan:
```text
Hybrid = α × SARIMA + (1 - α) × FTS_calibrated
```

### Evaluasi & Diagnosis
- Metrik **MAE, RMSE, MAPE, dan R²**.
- Perbandingan **SARIMA baseline, FTS terkalibrasi, dan Hybrid**.
- Ringkasan kontribusi/perubahan terhadap baseline.
- Error analysis dan residual diagnostics.
- **Uji Ljung-Box** untuk diagnosis autokorelasi residual.
- Tabel hasil testing dan detail prediksi.

### Forecast Masa Depan & Ekspor
- Forecast masa depan dengan pilihan **Window 6 → 1** atau **12 → 1**.
- Horizon forecast **1-24 periode**.
- Tabel forecast dan visualisasi historis + forecast.
- Download hasil historis, testing, tuning, dan forecast.
- **Complete Research Package** dalam bentuk ZIP untuk kebutuhan dokumentasi/arsip penelitian.

### Antarmuka
- Navigasi 9 tahap.
- **Plus Jakarta Sans** sebagai tipografi utama.
- **Bootstrap 5.3.3** untuk dukungan grid/layout responsif.
- **Bootstrap Icons 1.11.3** untuk ikon antarmuka.
- Desain antarmuka aplikasi berorientasi penelitian dengan deep navy, muted gold, white surface, spacing lega, dan card yang konsisten.
- Sidebar ringkas, page header bertahap, KPI cards, tabs, chart cards, dan responsive layout.

## 🔬 Alur Aplikasi

```text
Upload / Dataset Bawaan
        ↓
Pemeriksaan Struktur Data
        ↓
Preprocessing + Interpolasi Linear
        ↓
Pembagian Data 70:30 / 80:20
        ↓
Validation 12 Observasi
        ↓
SARIMA One-Step-Ahead
        +
FTS Chen Weighted
        ↓
Linear Calibration
        ↓
Tuning n_intervals + alpha
        ↓
Final Walk-Forward Testing
        ↓
Model Comparison
        ↓
Evaluasi MAE, RMSE, MAPE, R²
        ↓
Error / Residual Diagnostics
        ↓
Forecast Masa Depan
        ↓
Research Export
```

## 🧩 Konfigurasi Hybrid

Konfigurasi model yang digunakan pada aplikasi:

```text
Default SARIMA order       : (2,1,0)
Default seasonal order     : (1,1,0,12)
Default seasonal setting   : Aktif
Rentang p, d, q            : 0-5, 0-2, 0-5
Rentang P, D, Q            : 0-3, 0-2, 0-3
Rentang periode s          : 2-24 (default 12)
Validation size            : 12 observasi
FTS method                 : Chen Weighted
Window                     : 6 → 1 / 12 → 1
n_intervals                : 5-12
alpha                      : 0,80-0,99
```

Nilai di atas adalah **nilai awal dan rentang input**, bukan klaim bahwa satu konfigurasi SARIMA selalu menjadi konfigurasi terbaik. `p,d,q,P,D,Q,s` dipilih pengguna. Tuning internal yang dilakukan pada validation memilih konfigurasi FTS/ensemble (`n_intervals` dan `alpha`), bukan mencari otomatis semua kombinasi parameter SARIMA.

Formula ensemble:

```text
Hybrid = α × SARIMA + (1 − α) × FTS_calibrated
```

Pemilihan `n_intervals` dan `alpha` dilakukan pada **validation**, sedangkan testing digunakan sebagai evaluasi akhir.

### Cara Mengatur Parameter SARIMA

1. Buka menu **02 Input Dataset**.
2. Pada bagian **Pengaturan Parameter SARIMA**, ubah nilai `p`, `d`, `q`, `P`, `D`, `Q`, atau `s` sesuai skenario eksperimen. Untuk data bulanan dengan siklus tahunan, nilai awal `s = 12`.
3. Pilih apakah komponen musiman akan diaktifkan, lalu klik **Terapkan Parameter SARIMA**.
4. Klik **Jalankan Hybrid SARIMA-FTS** untuk menghitung ulang hasil. Jika konfigurasi berbeda dari hasil yang sedang ditampilkan, aplikasi memberi peringatan agar pipeline dijalankan ulang.
5. Periksa konfigurasi aktif pada menu **08 Konfigurasi Model** dan berkas konfigurasi pada ekspor penelitian.

Gunakan validation untuk memilih skenario model dan jangan memilih parameter hanya berdasarkan hasil testing, agar testing tetap menjadi evaluasi akhir.

## 📊 Menu Aplikasi

Aplikasi memiliki 9 menu:

```text
01  Beranda
02  Input Dataset
03  Data Historis
04  Hasil Forecasting
05  Windowing
06  Evaluasi Model
07  Forecast Masa Depan
08  Konfigurasi Model
09  Panduan Aplikasi
```

### 01 · Beranda
Ringkasan workspace penelitian, dataset aktif, workflow analisis, dan preview forecasting.

### 02 · Input Dataset
Upload dataset, pilih split 70:30 atau 80:20, atur parameter SARIMA non-musiman `(p,d,q)` dan musiman `(P,D,Q,s)`, terapkan pengaturan, jalankan pipeline, dan lihat audit preprocessing. Setelah parameter diubah, jalankan ulang pipeline agar hasil validation/testing dan proyeksi memakai konfigurasi terbaru.

### 03 · Data Historis
Eksplorasi rentang waktu, statistik dasar, grafik historis, tabel data, dan download.

### 04 · Hasil Forecasting
Melihat prediksi one-step-ahead dan membandingkan **Aktual, SARIMA, FTS terkalibrasi, dan Hybrid**.

### 05 · Windowing
Meninjau Window 6 → 1 dan Window 12 → 1 secara terpisah melalui tab agar informasi tetap fokus.

### 06 · Evaluasi Model
Menampilkan metrics testing, baseline comparison, model comparison, tuning visualization, error analysis, residual diagnostics, Ljung-Box, dan research export.

### 07 · Forecast Masa Depan
Membangun proyeksi setelah observasi historis terakhir dengan konfigurasi Window yang dipilih dan horizon 1-24 periode.

### 08 · Konfigurasi Model
Menampilkan parameter SARIMA, konfigurasi FTS, alpha, n_intervals, validation, dan ringkasan tuning.

### 09 · Panduan Aplikasi
Panduan penggunaan aplikasi, persiapan dataset, informasi program, metodologi, serta interpretasi metrik dan output.

## 🎨 Teknologi & Antarmuka

### Teknologi
- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Plotly
- Streamlit
- OpenPyXL / xlrd untuk input Excel

### UI/UX
- Plus Jakarta Sans
- Bootstrap 5.3.3 melalui CDN
- Bootstrap Icons 1.11.3 melalui CDN
- Responsive grid
- KPI cards
- Scenario tabs
- Research-oriented information hierarchy
- Luxury / clean / minimal visual system

Bootstrap dan Bootstrap Icons digunakan sebagai pendukung lapisan antarmuka; keduanya tidak mengubah proses perhitungan model.

## 📁 Struktur Repository

Struktur runtime minimum:

```text
Aplikasi-Peramalan-Hybrid-SARIMA-FTS-Waterpark-Sumenep-Asta-Tinggi/ 
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Validasi Akhir.py
└── Streamlit Data/
    └── data_historis.csv
```

File CSV hasil eksperimen seperti `hasil_testing.csv`, `window_6.csv`, `window_12.csv`, `metrics.csv`, dan `konfigurasi_model.csv` dapat disimpan sebagai **arsip/output tambahan**, tetapi `app.py` terbaru tidak mengunci ketujuh CSV tersebut sebagai dependency runtime.

Dataset `Streamlit Data/data_historis.csv` digunakan sebagai **dataset bawaan** sehingga aplikasi dapat dibuka tanpa upload terlebih dahulu.

## ▶️ Menjalankan Secara Lokal

Buka terminal pada folder repository:

```powershell
pip install -r requirements.txt
```

Validasi project:

```powershell
python "Validasi Akhir.py"
```

Setelah validasi berhasil:

```powershell
streamlit run app.py
```

Aplikasi dapat diakses melalui alamat lokal berikut:

```text
http://localhost:8501
```

## ☁️ Deployment Streamlit Community Cloud

Repository minimal:

```text
app.py
requirements.txt
README.md
Streamlit Data/data_historis.csv
```

Pada Streamlit Community Cloud:

```text
Repository       : repository GitHub Anda
Branch            : main
Main file path    : app.py
```

## 🧪 Catatan Metodologis

- Split dilakukan **secara kronologis**, bukan random shuffle.
- Validation berada sebelum testing dan digunakan untuk tuning.
- Parameter SARIMA `(p,d,q)` dan `(P,D,Q,s)` ditentukan melalui kontrol pengguna. Nilai awal aplikasi adalah `(2,1,0)` dan `(1,1,0,12)` dengan komponen musiman aktif.
- `n_intervals` dan `alpha` dipilih pada validation sebelum konfigurasi FTS/ensemble digunakan untuk testing; proses ini bukan optimasi otomatis seluruh parameter SARIMA.
- Forecast testing menggunakan pendekatan **one-step-ahead / walk-forward**.
- Nilai `0` dan missing yang berada pada posisi yang dapat diinterpolasi diperlakukan sebagai observasi yang perlu ditangani melalui interpolasi linear.
- Untuk dataset Asta Tinggi, tahun **2020 tidak tersedia** pada sumber dan tidak dibuat secara sintetis.
- Future forecast merupakan **proyeksi model**, bukan nilai aktual yang sudah terjadi.
- Audit preprocessing sebaiknya diperiksa sebelum interpretasi hasil.
- Metrik digunakan bersama grafik dan diagnostic agar interpretasi tidak hanya bergantung pada satu indikator.

## ✅ Validasi Project

Script `Validasi Akhir.py` disediakan untuk memeriksa konsistensi:

```text
Struktur Project
        ↓
app.py
        ↓
Konfigurasi Hybrid SARIMA-FTS
        ↓
Navigasi Aplikasi
        ↓
Dataset Bawaan
        ↓
README ↔ app.py
        ↓
requirements.txt
        ↓
Output CSV Opsional
        ↓
Notebook Opsional
```

Validator memeriksa default SARIMA, keberadaan kontrol parameter dinamis, jalur pemakaian konfigurasi pada pipeline dan future forecast, menu aplikasi, dataset bawaan, README, serta dependensi. Validator tidak mengunci satu angka MAE/RMSE/MAPE/R² karena hasil dapat berubah sesuai dataset, parameter yang dipilih, dan pembagian data pengguna.

## 👤 Project

**Aplikasi Peramalan Kunjungan Waterpark Sumenep (Asta Tinggi) Berbasis Streamlit menggunakan Metode Hybrid SARIMA-FTS**

Bagian dari kegiatan:

**MBKM Skema Penelitian/Riset**

Topik penelitian:

**Implementasi Hybrid Metode SARIMA-FTS Untuk Peramalan Kunjungan Waterpark Sumenep (Asta Tinggi)**

Aplikasi ini digunakan sebagai media implementasi dan dokumentasi eksperimen **Hybrid SARIMA-FTS** pada data kunjungan Waterpark Sumenep (Asta Tinggi).

## 📄 Lisensi

Project digunakan untuk kebutuhan akademik, dokumentasi penelitian, dan demonstrasi implementasi.
