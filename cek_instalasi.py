"""Pemeriksa instalasi. Jalankan:  python cek_instalasi.py
Hasilnya menjelaskan apa yang kurang. Kalau ada masalah, salin seluruh keluarannya dan kirim."""
import sys, platform, importlib, uuid
from pathlib import Path

here, bad = Path(__file__).resolve().parent, []


def ok(msg): print(f"  [OK]    {msg}")
def no(msg, fix): print(f"  [GAGAL] {msg}\n          -> {fix}"); bad.append(msg)


print(f"Python {platform.python_version()} · {platform.system()} {platform.machine()}")
if sys.version_info < (3, 10): no("Python terlalu lama", "pasang Python 3.10 – 3.12 dari python.org")
elif sys.version_info >= (3, 14): print("  [CATATAN] Python 3.14+ kadang belum punya paket onnx/onnxruntime. Jika pemasangan gagal, pakai Python 3.12.")
else: ok("Versi Python cocok")
print("\nFile aplikasi:")
for f in ("main.py", "feature_engine.py", "ml_onnx_trainer.py", "package_export.py", "styles.py", "home_view.py", "radar_map_view.py", "training_studio_view.py", "orders_log_view.py", "requirements.txt"):
    (ok if (here / f).exists() else lambda m: no(m, "unduh ulang paket zip lengkap"))(f if (here / f).exists() else f"{f} TIDAK ADA")
print("\nLibrary:")
for mod in ("numpy", "onnx", "onnxruntime", "PySide6", "PySide6.QtWebEngineWidgets"):
    try: m = importlib.import_module(mod); ok(f"{mod} {getattr(m, '__version__', '')}".strip())
    except Exception as e: no(f"{mod}: {type(e).__name__}: {e}"[:150], "pip install -r requirements.txt  (di folder ini, dengan lingkungan Python yang sama)")
print("\nFolder bisa ditulis:")
try: (here / "models").mkdir(exist_ok=True); t = here / "models" / ".tes"; t.write_text("x"); t.unlink(); ok("folder models/ bisa dibuat & ditulis")
except Exception as e: no(f"tidak bisa menulis di {here}: {e}", "pindahkan folder ke Documents/Dokumen (bukan Program Files / dalam file zip)")
if not bad:
    print("\nUji pelatihan cepat (±5 detik)…")
    history_path = here / "models" / "history.json"
    history_backup = history_path.read_bytes() if history_path.exists() else None
    test_name = "_cek_" + uuid.uuid4().hex
    try:
        sys.path.insert(0, str(here)); import feature_engine as fe, ml_onnx_trainer as tr
        st = []; fe.merge_orders(st, fe.demo_orders(days=20)); r = tr.train_model(st, dict(name=test_name, algo="glm", cv=False, epochs=5))
        tr.Runtime(r["onnx_path"]).predict([(0, 0)], [__import__("datetime").datetime.now()]); ok("latih → ekspor ONNX → prediksi berfungsi")
    except Exception as e: import traceback; traceback.print_exc(); no(f"uji pelatihan gagal: {e}"[:150], "salin keluaran di atas dan kirim")
    finally:
        try:
            import ml_onnx_trainer as tr
            for p in tr.MODELS.glob(f"{test_name}*"): p.unlink(missing_ok=True)
            if history_backup is None: history_path.unlink(missing_ok=True)
            else: history_path.write_bytes(history_backup)
        except Exception as e: no(f"pembersihan uji sementara gagal: {e}"[:150], "periksa folder models/ dan pulihkan history.json dari cadangan bila perlu")
print("\n" + ("SEMUA OK. Jalankan:  python main.py" if not bad else f"ADA {len(bad)} MASALAH. Perbaiki yang bertanda [GAGAL], lalu jalankan ulang file ini."))
sys.exit(1 if bad else 0)
