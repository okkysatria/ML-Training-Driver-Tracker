import sys, json, shutil
from pathlib import Path


def _guard():
    """Periksa dulu: versi Python, file lengkap, library terpasang. Pesan error dibuat jelas untuk pemula."""
    here, problems = Path(__file__).resolve().parent, []
    if sys.version_info < (3, 10): problems.append(f"Python {sys.version.split()[0]} terlalu lama. Pakai Python 3.10 – 3.12.")
    need = ("feature_engine.py", "ml_onnx_trainer.py", "package_export.py", "styles.py", "home_view.py", "radar_map_view.py", "training_studio_view.py", "orders_log_view.py")
    miss = [f for f in need if not (here / f).exists()]
    if miss: problems.append("File ini TIDAK ADA di folder: " + ", ".join(miss) + "\n   -> Unduh ulang paket lengkap (driver_tracker.zip) dan ekstrak SEMUANYA ke satu folder.")
    for mod in ("PySide6", "numpy", "onnx", "onnxruntime"):
        try: __import__(mod)
        except Exception as e: problems.append(f"Library '{mod}' belum terpasang atau rusak ({type(e).__name__}).\n   -> Jalankan: pip install -r requirements.txt")
    if problems:
        print("\n=== APLIKASI BELUM BISA DIJALANKAN ===\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(problems, 1)) + "\n\nUntuk pemeriksaan lengkap jalankan: python cek_instalasi.py")
        try: input("\nTekan Enter untuk menutup…")
        except Exception: pass
        sys.exit(1)


_guard()
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QPushButton, QStackedWidget, QLabel, QMessageBox, QSizePolicy
import ml_onnx_trainer as tr
from styles import QSS
from home_view import HomeView
from radar_map_view import RadarView
from training_studio_view import StudioView
from orders_log_view import OrdersLogView

STORE = Path(__file__).parent / "orders_data.json"  # semua order dari JSON/CSV yang diimpor, permanen


class Main(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("ML Training Driver Tracker — ML Demand Training & ONNX Radar"); self.cfg = QSettings("MLTrainingDriverTracker", "App")
        g = self.cfg.value("geometry")
        if g: self.restoreGeometry(g)
        else: self.resize(1360, 860)
        try: self.orders = json.loads(STORE.read_text("utf-8")) if STORE.exists() else []
        except Exception: self.orders = []
        self.home, self.radar, self.studio, self.log = HomeView(), RadarView(self.orders), StudioView(self.orders), OrdersLogView(self.orders)
        self.stack = QStackedWidget()
        for w in (self.home, self.radar, self.studio, self.log): self.stack.addWidget(w)  # indeks 0..3
        side = QFrame(); side.setObjectName("sidebar"); side.setFixedWidth(210); sv = QVBoxLayout(side); sv.setContentsMargins(8, 0, 8, 12)
        # ── Brand / Logo Widget ──────────────────────────────────────────────
        brand_w = QWidget(); brand_w.setObjectName("brandWidget"); brand_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        brand_v = QVBoxLayout(brand_w); brand_v.setContentsMargins(10, 14, 10, 10); brand_v.setSpacing(4)
        _logo_path = str(Path(__file__).parent / "assets" / "logo_128.png")
        logo_lbl = QLabel(); logo_lbl.setObjectName("brandLogo"); logo_lbl.setFixedSize(48, 48); logo_lbl.setAlignment(Qt.AlignCenter)
        _pm = QPixmap(_logo_path)
        if not _pm.isNull(): logo_lbl.setPixmap(_pm.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else: logo_lbl.setText("📡")
        brand_title = QLabel("Driver Tracker"); brand_title.setObjectName("brandTitle")
        brand_sub = QLabel("ML · ONNX Radar"); brand_sub.setObjectName("brandSubtitle")
        brand_v.addWidget(logo_lbl); brand_v.addWidget(brand_title); brand_v.addWidget(brand_sub)
        sv.addWidget(brand_w)
        # Set app window icon
        _icon_path = str(Path(__file__).parent / "assets" / "app_icon.ico")
        _icon_pm = QPixmap(_icon_path)
        if not _icon_pm.isNull(): self.setWindowIcon(QIcon(_icon_pm))
        self.btns = []
        for i, n in enumerate(("🏠  Mulai", "📡  Peta Radar", "🧠  Studio Pelatihan", "📋  Riwayat Order")):
            b = QPushButton(n); b.setCheckable(True); b.clicked.connect(lambda _, i=i: self.go(i)); sv.addWidget(b); self.btns.append(b)
        sv.addStretch(); self.st_data, self.st_model = QLabel(), QLabel(); self.st_model.setWordWrap(True)
        for l in (self.st_data, self.st_model): l.setObjectName("sidestat"); sv.addWidget(l)
        c = QWidget(); h = QHBoxLayout(c); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0); h.addWidget(side); h.addWidget(self.stack, 1); self.setCentralWidget(c)
        self.home.act.connect(self.act); self.radar.train_requested.connect(lambda: self.go(2)); self.radar.order_logged.connect(self.changed)
        self.radar.model_loaded.connect(self.model_changed); self.studio.model_ready.connect(self.radar.load_model); self.studio.go_radar.connect(lambda: self.go(1))
        self.studio.orders_changed.connect(self.changed); self.log.changed.connect(self.changed)
        for v in (self.studio, self.radar): v.notify.connect(lambda m: self.statusBar().showMessage(m, 8000))
        p = tr.latest_model()
        if p:
            try: self.radar.load_model(p)
            except Exception as e: QMessageBox.warning(self, "Model", f"Gagal memuat model terakhir:\n{e}")
        self.update_status(); self.refresh_home(); self.go(0)

    def go(self, i):
        self.stack.setCurrentIndex(i)
        for k, b in enumerate(self.btns): b.setChecked(k == i)

    def act(self, a):
        if a == "import": self.go(2); self.studio.upload()
        else: self.go(2 if a == "train" else 1)

    def update_status(self):
        self.st_data.setText(f"📦 {len(self.orders)} order tersimpan"); m = self.radar.rt.meta if self.radar.rt else None
        self.st_model.setText(f"🧠 {m['name']} · {tr.quality_label(m['metrics'])[0]}" if m else "🧠 Belum ada model")

    def refresh_home(self): self.home.refresh(self.orders, self.radar.rt.meta if self.radar.rt else None)

    def model_changed(self, p): self.studio.set_active(p); self.update_status(); self.refresh_home()

    def changed(self):  # satu pintu: simpan -> sinkronkan semua halaman
        if STORE.exists(): shutil.copy(STORE, STORE.with_suffix(".bak"))  # cadangan otomatis
        STORE.write_text(json.dumps(self.orders, ensure_ascii=False), "utf-8")
        self.log.reload(); self.studio.refresh_report(); self.radar.refresh(); self.refresh_home(); self.update_status()

    def closeEvent(self, e): self.cfg.setValue("geometry", self.saveGeometry()); super().closeEvent(e)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv); app.setStyleSheet(QSS); w = Main(); w.show(); code = app.exec()
    except Exception:
        import traceback
        msg = traceback.format_exc(); (Path(__file__).resolve().parent / "error_log.txt").write_text(msg, "utf-8")
        print("\n=== ADA ERROR SAAT MEMBUKA APLIKASI ===\n" + msg + "\n(Disimpan juga di error_log.txt. Kirim isinya ke saya.)")
        try: input("Tekan Enter untuk menutup…")
        except Exception: pass
        sys.exit(1)
    sys.exit(code)
