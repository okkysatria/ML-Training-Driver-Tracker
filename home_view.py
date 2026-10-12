from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame, QLabel, QPushButton, QScrollArea
import feature_engine as fe
import ml_onnx_trainer as tr


class HomeView(QScrollArea):
    act = Signal(str)  # "import" | "train" | "map"

    def __init__(self):
        super().__init__(); self.setWidgetResizable(True); w = QWidget(); w.setObjectName("page"); self.setWidget(w)
        v = QVBoxLayout(w); v.setContentsMargins(24, 18, 24, 18); v.setSpacing(14); self._a = "import"
        h = QLabel("ML Training Driver Tracker"); h.setObjectName("h1"); d = QLabel("Belajar dari riwayat order Anda untuk menebak area yang ramai 60 menit ke depan."); d.setObjectName("hint")
        v.addWidget(h); v.addWidget(d)
        hero = QFrame(); hero.setObjectName("card"); hero.setStyleSheet("QFrame#card { border: 1.5px solid #059669; }"); hv = QVBoxLayout(hero); hv.setContentsMargins(20, 16, 20, 18)
        k = QLabel("LANGKAH BERIKUTNYA"); k.setObjectName("hint"); self.ht = QLabel(); self.ht.setObjectName("h1")
        self.hd = QLabel(); self.hd.setWordWrap(True); self.hd.setObjectName("muted"); self.hb = QPushButton(); self.hb.setObjectName("primary")
        self.hb.clicked.connect(lambda: self.act.emit(self._a))
        for x in (k, self.ht, self.hd, self.hb): hv.addWidget(x)
        v.addWidget(hero)
        c = QFrame(); c.setObjectName("card"); cv = QVBoxLayout(c); cv.setContentsMargins(20, 14, 20, 14); t = QLabel("Progres"); t.setObjectName("title"); cv.addWidget(t)
        self.rows = [QLabel() for _ in range(3)]
        for r in self.rows: r.setWordWrap(True); cv.addWidget(r)
        v.addWidget(c)
        c = QFrame(); c.setObjectName("card"); cv = QVBoxLayout(c); cv.setContentsMargins(20, 14, 20, 14); t = QLabel("Rutinitas supaya makin pintar"); t.setObjectName("title"); cv.addWidget(t)
        s = QLabel("• Saat narik: klik \"+ Catat Order di Pin\" di Peta Radar tiap dapat order.\n• Tiap bulan: Studio Pelatihan → ⟳ Update Bulanan (JSON baru digabung dengan SEMUA data lama, lalu dilatih ulang).\n• Mau dipakai aplikasi lain: Studio Pelatihan → langkah ③ → Uji Paket → Ekspor Paket (.zip).")
        s.setWordWrap(True); s.setObjectName("muted"); cv.addWidget(s); v.addWidget(c)
        c = QFrame(); c.setObjectName("card"); cv = QVBoxLayout(c); cv.setContentsMargins(20, 14, 20, 14); t = QLabel("Riwayat pertumbuhan model"); t.setObjectName("title")
        self.hist = QLabel(); self.hist.setObjectName("mono"); self.hist.setWordWrap(True); cv.addWidget(t); cv.addWidget(self.hist); v.addWidget(c); v.addStretch()

    def refresh(self, store, meta):
        r = fe.data_report(store); ok = r["ready"]; trained = meta.get("n_orders", meta["n_train"] + meta["n_test"]) if meta else 0
        q = tr.quality_label(meta["metrics"])[0] if meta else ""
        if not ok: self._a, t, d, b = "import", "Impor data order Anda", f"Anda punya {r['n']} order valid dalam {r['days']} hari. Butuh minimal 100 order dan 7 hari berbeda (file JSON atau CSV, boleh banyak sekaligus).", "Impor JSON / CSV…"
        elif not meta: self._a, t, d, b = "train", "Latih model pertama Anda", "Data sudah cukup. Satu klik di Studio Pelatihan dan model siap dalam beberapa detik.", "Buka Studio Pelatihan"
        elif r["n"] >= 1.2 * trained + 10: self._a, t, d, b = "train", "Data bertambah, saatnya latih ulang", f"Ada {r['n'] - trained} order baru sejak model terakhir dilatih. Latih ulang agar prediksi mengikuti pola terbaru.", "Buka Studio Pelatihan"
        else: self._a, t, d, b = "map", "Pakai di Peta Radar", f"Model aktif ({meta['name']}, kualitas {q}) siap. Pilih posisi, atur jam, dan lihat area yang disarankan.", "Buka Peta Radar"
        self.ht.setText(t); self.hd.setText(d); self.hb.setText(b)
        self.rows[0].setText(f"{'✔' if ok else '○'}  Data: {r['n']} order · {r['days']} hari · {r['areas']} area")
        self.rows[1].setText(f"✔  Model aktif: {meta['name']} (kualitas {q})" if meta else "○  Belum ada model")
        self.rows[2].setText("✔  Siap dipakai di Peta Radar" if meta else "○  Peta Radar butuh model dulu")
        hs = tr.load_history()[-6:][::-1]
        self.hist.setText("\n".join(f"{e['created'][:10]} · {e['n_orders']} order ({e['date_from']} → {e['date_to']}) · {e['quality']} · Top-5 {e['top5']:.0%} vs tebakan {e['baseline']:.0%}" + (" · lanjutan" if e['warm_start'] else "") for e in hs)
                      + ("\n(Antar-model tidak persis sebanding karena periode ujinya berbeda.)" if hs else "Belum ada model."))
