from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Semua path dihitung relatif terhadap lokasi skrip agar validator dapat
# dijalankan dari direktori repository maupun melalui path absolut.
ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
README = ROOT / "README.md"
REQUIREMENTS = ROOT / "requirements.txt"
DATA_DIR = ROOT / "Streamlit Data"
DATA_FILE = DATA_DIR / "data_historis.csv"

MENU = [
    "01  Beranda",
    "02  Input Dataset",
    "03  Data Historis",
    "04  Hasil Forecasting",
    "05  Windowing",
    "06  Evaluasi Model",
    "07  Forecast Masa Depan",
    "08  Konfigurasi Model",
    "09  Panduan Aplikasi",
]
OPTIONAL_CSV = [
    "hasil_training.csv",
    "hasil_testing.csv",
    "window_6.csv",
    "window_12.csv",
    "metrics.csv",
    "konfigurasi_model.csv",
]

checks: list[bool] = []
info: list[tuple[str, str]] = []


def status(label: str, ok: bool, detail: str = "") -> None:
    checks.append(bool(ok))
    print(f"{label:<72}: {'OK' if ok else 'GAGAL'}" + (f" — {detail}" if detail else ""))


def note(label: str, detail: str = "") -> None:
    info.append((label, detail))
    print(f"{label:<72}: INFO" + (f" — {detail}" if detail else ""))


def assignment_value(tree: ast.AST | None, name: str):
    """Get a simple module-level assigned value using ast.literal_eval."""
    if tree is None:
        return None
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                try:
                    return ast.literal_eval(value_node)
                except (ValueError, TypeError):
                    return None
    return None


print("=" * 100)
print("Validasi Aplikasi Peramalan Hybrid SARIMA–FTS · Waterpark Sumenep Asta Tinggi")
print(f"Folder project: {ROOT}")
print("=" * 100)

# 1. PROJECT STRUCTURE
print("\n[1] STRUKTUR PROJECT")
status("app.py tersedia", APP.is_file())
status("README.md tersedia", README.is_file())
status("requirements.txt tersedia", REQUIREMENTS.is_file())
status("Folder Streamlit Data tersedia", DATA_DIR.is_dir())
status("Dataset bawaan data_historis.csv tersedia", DATA_FILE.is_file())

# 2. APP SYNTAX AND FUNCTIONS
print("\n[2] SOURCE CODE app.py")
app_text = APP.read_text(encoding="utf-8", errors="ignore") if APP.is_file() else ""
tree = None
if app_text:
    try:
        tree = ast.parse(app_text)
        status("app.py bebas syntax error", True)
    except SyntaxError as exc:
        status("app.py bebas syntax error", False, f"baris {exc.lineno}: {exc.msg}")
else:
    status("app.py bebas syntax error", False, "app.py tidak ditemukan atau kosong")
page_title_match = re.search(r'page_title\s*=\s*(["\'])(.*?)\1', app_text)
page_title_text = page_title_match.group(2) if page_title_match else ""
status("Nama produk aplikasi tercantum pada app.py", "Aplikasi Peramalan Hybrid SARIMA–FTS" in app_text)
status("Judul produk di app.py menggunakan penamaan Aplikasi", bool(page_title_text) and "Aplikasi Peramalan" in page_title_text and "Dashboard" not in page_title_text, page_title_text)

required_functions = [
    "normalize_model_config",
    "effective_seasonal_order",
    "model_config_label",
    "fit_sarima_forecast",
    "fts_chen_forecast",
    "run_hybrid_pipeline",
    "cached_run_pipeline",
    "make_future_forecast",
    "evaluate_metrics",
]
for fn in required_functions:
    status(f"Fungsi {fn} tersedia", bool(re.search(rf"def\s+{re.escape(fn)}\s*\(", app_text)))

# 3. DYNAMIC SARIMA CONFIGURATION
print("\n[3] KONFIGURASI SARIMA DINAMIS")
default_cfg = assignment_value(tree, "DEFAULT_MODEL_CONFIG")
expected_cfg = {
    "order": (2, 1, 0),
    "seasonal_order": (1, 1, 0, 12),
    "seasonal_enabled": True,
}
status("Default parameter SARIMA sesuai konfigurasi awal", default_cfg == expected_cfg, f"terbaca {default_cfg}")
status("Parser/normalisasi konfigurasi tersedia", "def normalize_model_config(" in app_text)
status("Order musiman efektif mengikuti toggle", 'return cfg["seasonal_order"] if cfg["seasonal_enabled"] else (0, 0, 0, 0)' in app_text)
status("Estimator SARIMA menerima model_config", bool(re.search(r"def\s+fit_sarima_forecast\s*\(\s*history\s*,\s*model_config\s*\)", app_text)))
status("Pipeline menerima model_config", bool(re.search(r"def\s+run_hybrid_pipeline\s*\([^\n]*model_config\s*=\s*None", app_text)))
status("Cache pipeline memperhitungkan model_config", bool(re.search(r"def\s+cached_run_pipeline\s*\([^\n]*model_config", app_text)))
status("Future forecast memakai konfigurasi tersimpan", 'normalize_model_config(bundle.get("model_config"))' in app_text and 'seasonal_order=effective_seasonal_order(model_config)' in app_text)
status("Ekspor mencatat order SARIMA pilihan", '"SARIMA order (p,d,q):' in app_text and '"SARIMA seasonal order (P,D,Q,s):' in app_text)

# Check parameter controls and configured bounds.
parameter_controls = [
    ("Kontrol p rentang 0–5", r'st\.number_input\("p · AR",\s*min_value=0,\s*max_value=5'),
    ("Kontrol d rentang 0–2", r'st\.number_input\("d · Differencing",\s*min_value=0,\s*max_value=2'),
    ("Kontrol q rentang 0–5", r'st\.number_input\("q · MA",\s*min_value=0,\s*max_value=5'),
    ("Kontrol P rentang 0–3", r'st\.number_input\("P · Seasonal AR",\s*min_value=0,\s*max_value=3'),
    ("Kontrol D rentang 0–2", r'st\.number_input\("D · Seasonal differencing",\s*min_value=0,\s*max_value=2'),
    ("Kontrol Q rentang 0–3", r'st\.number_input\("Q · Seasonal MA",\s*min_value=0,\s*max_value=3'),
    ("Kontrol s rentang 2–24", r'st\.number_input\("s · Periode musiman",\s*min_value=2,\s*max_value=24'),
]
for label, pattern in parameter_controls:
    status(label, bool(re.search(pattern, app_text)))
status("Toggle komponen musiman tersedia", '"Aktifkan komponen musiman"' in app_text and '"seasonal_enabled": bool(user_seasonal_enabled)' in app_text)
status("Tombol penerapan parameter tersedia", 'st.form_submit_button("Terapkan Parameter SARIMA"' in app_text)
status("Peringatan jika pipeline belum memakai konfigurasi terbaru", "Parameter tersimpan belum diterapkan" in app_text)
status("Konfigurasi pilihan diteruskan saat pipeline dijalankan", "cached_run_pipeline(file_bytes, filename, split_ratio, active_model_config)" in app_text)

# Core method and model settings.
print("\n[4] METODOLOGI HYBRID SARIMA–FTS")
status("Seasonal period default = 12", assignment_value(tree, "SEASONAL_PERIOD") == 12)
status("Validation size = 12", assignment_value(tree, "VALIDATION_SIZE") == 12)
status("Window 6 dan 12 tersedia", assignment_value(tree, "WINDOWS") == [6, 12])
status("FTS interval grid 5–12 tersedia", "list(range(5, 13))" in app_text or "range(5, 13)" in app_text)
status("Alpha grid 0,80–0,99 tersedia", "np.arange(0.80, 1.00, 0.01)" in app_text)
status("FTS Chen Weighted tersedia", "def fts_chen_forecast" in app_text and "weighted=True" in app_text)
status("Linear Regression calibration tersedia", "LinearRegression" in app_text)
status("Formula Hybrid tersedia", "alpha * p_s + (1 - alpha) * p_f" in app_text)
status("Walk-forward testing tersedia", "for j, t in enumerate(range(train_n, n))" in app_text)
status("Future forecast tersedia", "def make_future_forecast" in app_text)
status("Preprocessing menangani nol dan missing dengan interpolasi", "replace(0, np.nan).interpolate" in app_text)

# 5. NAVIGATION
print("\n[5] NAVIGASI APLIKASI")
menu_match = re.search(r"menu_options\s*=\s*(\[[\s\S]*?\])\s*\nmenu\s*=", app_text)
detected = []
if menu_match:
    try:
        detected = ast.literal_eval(menu_match.group(1))
    except (SyntaxError, ValueError):
        detected = []
status("9 menu utama sesuai app.py", detected == MENU, f"terbaca {detected}")
for item in MENU:
    status(f"Menu {item} tersedia", item in detected)
status("Sidebar navigation tersedia", "st.sidebar.radio" in app_text)
status("Halaman konfigurasi memaparkan parameter aktif", '"Parameter":"Order non-musiman (p,d,q)"' in app_text and '"Parameter":"Order musiman (P,D,Q,s)"' in app_text)

# 6. DEFAULT DATASET
print("\n[6] DATASET BAWAAN")
if DATA_FILE.is_file():
    try:
        df = pd.read_csv(DATA_FILE)
        status("data_historis.csv dapat dibaca", True, f"{len(df)} baris")
        status("Jumlah observasi = 108", len(df) == 108, f"terbaca {len(df)}")
        status("Kolom Tanggal tersedia", "Tanggal" in df.columns)
        status("Kolom Jumlah_Kunjungan tersedia", "Jumlah_Kunjungan" in df.columns)
        if {"Tanggal", "Jumlah_Kunjungan"}.issubset(df.columns):
            dt = pd.to_datetime(df["Tanggal"], errors="coerce")
            raw_vals = df["Jumlah_Kunjungan"]
            vals = pd.to_numeric(raw_vals, errors="coerce")
            # Missing/blank values are allowed: preprocessing handles them.
            # Non-empty strings that fail numeric conversion are not allowed.
            raw_text = raw_vals.astype("string").str.strip()
            invalid_numeric = vals.isna() & raw_vals.notna() & raw_text.ne("")
            status("Semua tanggal dapat dikonversi", bool(dt.notna().all()))
            status("Tidak ada duplikasi tanggal", int(dt.duplicated().sum()) == 0)
            status("Tanggal awal = 2015-01-01", dt.min() == pd.Timestamp("2015-01-01"), str(dt.min()))
            status("Tanggal akhir = 2024-12-01", dt.max() == pd.Timestamp("2024-12-01"), str(dt.max()))
            status("Tahun 2020 tidak dibuat sintetis", 2020 not in set(dt.dt.year))
            status("Nilai kunjungan non-kosong dapat dikonversi numerik", int(invalid_numeric.sum()) == 0, f"nilai non-numerik={int(invalid_numeric.sum())}")
            status("Dataset bawaan tidak memiliki nilai 0", int((vals == 0).sum()) == 0)
            note("Status nilai dataset bawaan", f"missing={int(vals.isna().sum())}, zero={int((vals == 0).sum())}; missing diperbolehkan untuk diproses aplikasi.")
    except Exception as exc:
        status("data_historis.csv dapat dibaca", False, str(exc))
else:
    note("Pemeriksaan isi dataset dilewati", "Dataset bawaan tidak ditemukan pada lokasi yang diharapkan.")

# 7. README CONSISTENCY
print("\n[7] KONSISTENSI README ↔ app.py")
readme_text = README.read_text(encoding="utf-8", errors="ignore") if README.is_file() else ""
readme_checks = [
    ("README menyebut Hybrid SARIMA–FTS", "Hybrid SARIMA–FTS"),
    ("README menyebut split 70:30", "70:30"),
    ("README menyebut split 80:20", "80:20"),
    ("README menyebut default order SARIMA", "(2,1,0)"),
    ("README menyebut default seasonal order", "(1,1,0,12)"),
    ("README menjelaskan parameter SARIMA dinamis", "Parameter SARIMA dinamis"),
    ("README menjelaskan komponen musiman", "Komponen musiman"),
    ("README menyebut FTS Chen Weighted", "FTS:** Chen Weighted"),
    ("README menyebut Window 6 → 1", "6 → 1"),
    ("README menyebut Window 12 → 1", "12 → 1"),
    ("README menyebut MAE", "MAE"),
    ("README menyebut RMSE", "RMSE"),
    ("README menyebut MAPE", "MAPE"),
    ("README menyebut R²", "R²"),
    ("README menyebut Ljung–Box", "Ljung–Box"),
    ("README menyebut Complete Research Package", "Complete Research Package"),
    ("README menyebut perintah validator yang benar", 'python "Validasi Akhir.py"'),
    ("README menyebut Panduan Aplikasi", "Panduan Aplikasi"),
]
for label, token in readme_checks:
    status(label, token in readme_text)
status("README memakai nama produk Aplikasi Peramalan", "Aplikasi Peramalan Kunjungan Waterpark Sumenep (Asta Tinggi)" in readme_text)
status("README tidak menggunakan nama produk lama", "Forecast Dashboard" not in readme_text and "Hybrid SARIMA–FTS Forecast Dashboard" not in readme_text)
for item in MENU:
    status(f"README menyebut {item}", item in readme_text)

# 8. REQUIREMENTS
print("\n[8] REQUIREMENTS")
req_lines = []
if REQUIREMENTS.is_file():
    req_lines = [line.strip().lower() for line in REQUIREMENTS.read_text(encoding="utf-8", errors="ignore").splitlines()]
req_normalized = {re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip() for line in req_lines if line and not line.startswith("#")}
for pkg in ["streamlit", "pandas", "numpy", "plotly", "statsmodels", "scikit-learn", "openpyxl", "xlrd"]:
    status(f"requirements memuat {pkg}", pkg in req_normalized)

# 9. OPTIONAL CSV ARCHIVES
print("\n[9] OUTPUT CSV STATIS / ARSIP")
for name in OPTIONAL_CSV:
    path = DATA_DIR / name
    if path.is_file():
        try:
            out_df = pd.read_csv(path)
            note(f"{name} tersedia", f"{len(out_df)} baris")
        except Exception as exc:
            status(f"{name} dapat dibaca", False, str(exc))
    else:
        note(f"{name} tidak tersedia", "Opsional; aplikasi menghitung ulang hasil secara dinamis.")

# 10. UI AND DEPLOYMENT
print("\n[10] UI / DEPLOYMENT")
for label, token in [
    ("Plus Jakarta Sans tersedia", "Plus Jakarta Sans"),
    ("Bootstrap 5.3.3 tersedia", "bootstrap@5.3.3"),
    ("Bootstrap Icons 1.11.3 tersedia", "bootstrap-icons@1.11.3"),
    ("Responsive grid tersedia", "grid-template-columns"),
    ("Research package export tersedia", "make_research_zip"),
    ("Dataset default diarahkan ke Streamlit Data", 'Path("Streamlit Data")'),
    ("Upload dataset tersedia", "st.file_uploader"),
]:
    status(label, token in app_text)
status("README menjadikan app.py sebagai entry point", "streamlit run app.py" in readme_text)
status("README menjadikan app.py sebagai main file deployment", "Main file path" in readme_text and "app.py" in readme_text)
status("README mencantumkan alur deploy Streamlit", "Streamlit Community Cloud" in readme_text)

# 11. NOTEBOOKS ARE OPTIONAL
print("\n[11] NOTEBOOK / ARSIP EKSPERIMEN")
notebooks = list(ROOT.glob("*.ipynb"))
if notebooks:
    note("Notebook tersedia", f"{len(notebooks)} file; notebook tidak wajib untuk runtime aplikasi.")
else:
    note("Notebook tidak ditemukan", "Opsional untuk runtime aplikasi; tambahkan bila diperlukan untuk arsip eksperimen.")

# FINAL REPORT
print("\n" + "=" * 100)
passed = sum(checks)
total = len(checks)
failed = total - passed
print("STATUS AKHIR")
print(f"Pemeriksaan: {total} | Berhasil: {passed} | Gagal: {failed} | Informasi: {len(info)}")
print("Kesimpulan:", "LULUS" if failed == 0 else "BELUM LULUS")
if info:
    print("\nCATATAN INFORMASI")
    for label, detail in info:
        print(f"- {label}" + (f": {detail}" if detail else ""))
print("=" * 100)
