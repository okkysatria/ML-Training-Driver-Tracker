import json, shutil
from collections import Counter
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QToolButton, QMenu, QComboBox, QProgressBar,
                               QFileDialog, QCheckBox, QSpinBox, QFormLayout, QScrollArea, QMessageBox, QInputDialog)
import feature_engine as fe
import ml_onnx_trainer as tr
import package_export as pk
from orders_log_view import import_json_dialog


def chip(lbl, kind, text):
    lbl.setText(text); lbl.setProperty("chip", kind); lbl.style().unpolish(lbl); lbl.style().polish(lbl)


def tile(caption):
    f = QFrame(); f.setObjectName("tile"); v = QVBoxLayout(f); v.setContentsMargins(12, 8, 12, 8); v.setSpacing(0)
    a = QLabel("–"); a.setObjectName("kpi"); b = QLabel(caption); b.setObjectName("hint"); b.setWordWrap(True); v.addWidget(a); v.addWidget(b)
    return f, a, b


def card(num, title):
    f = QFrame(); f.setObjectName("card"); v = QVBoxLayout(f); v.setContentsMargins(18, 14, 18, 16); v.setSpacing(10)
    h = QLabel(f"{num}  {title}"); h.setObjectName("title"); v.addWidget(h)
    return f, v


class Worker(QThread):
    prog = Signal(int, str); done = Signal(dict); fail = Signal(str)

    def __init__(self, orders, cfg): super().__init__(); self.o, self.c, self.stop = orders, cfg, False

    def run(self):
        def cb(p, m):
            if self.stop: raise InterruptedError()
            self.prog.emit(int(p), m)
        try: self.done.emit(tr.train_model(self.o, self.c, cb))
        except InterruptedError: self.fail.emit("Pelatihan dibatalkan.")
        except Exception as e: self.fail.emit(str(e))


class PkgWorker(QThread):  # uji paket jalan di thread terpisah agar UI tidak membeku
    done = Signal(dict)

    def __init__(self, p): super().__init__(); self.p = p

    def run(self):
        try: self.done.emit(pk.test_package(self.p))
        except Exception as e: self.done.emit(dict(ok=False, report=f"✖ Uji gagal dijalankan: {e}"))


class StudioView(QScrollArea):
    model_ready = Signal(str)
    go_radar = Signal()
    orders_changed = Signal()
    notify = Signal(str)

    def __init__(self, store):
        super().__init__(); self.setWidgetResizable(True); self.store, self.active, self.result, self.training, self._then = store, None, None, False, False
        self.qs = QSettings("MLTrainingDriverTracker", "Studio"); w = QWidget(); w.setObjectName("page"); self.setWidget(w)
        v = QVBoxLayout(w); v.setContentsMargins(24, 18, 24, 18); v.setSpacing(14)
        h = QLabel("Studio Pelatihan"); h.setObjectName("h1"); s = QLabel("Tiga langkah: siapkan data → latih → uji & bagikan. Pengaturan bawaan sudah cocok untuk pemula."); s.setObjectName("hint"); v.addWidget(h); v.addWidget(s)
        # ① DATA
        c1, v1 = card("①", "Data order"); v.addWidget(c1); tr1 = QHBoxLayout(); v1.addLayout(tr1)
        self.t_n, self.t_d, self.t_a = tile("order valid"), tile("hari berbeda"), tile("area (grid)")
        for t in (self.t_n, self.t_d, self.t_a): tr1.addWidget(t[0])
        self.chip_ready = QLabel(); tr1.addWidget(self.chip_ready, 0, Qt.AlignVCenter); tr1.addStretch()
        self.tips = QLabel(); self.tips.setObjectName("hint"); self.tips.setWordWrap(True); v1.addWidget(self.tips)
        row = QHBoxLayout(); v1.addLayout(row)
        b = QPushButton("Impor banyak JSON / CSV…"); b.setObjectName("primary"); b.setToolTip("Pilih banyak file sekaligus dengan Ctrl/Shift. Semua file digabung; duplikat dibuang."); b.clicked.connect(lambda: self.upload()); row.addWidget(b)
        b = QPushButton("⟳ Update Bulanan"); b.setToolTip("Impor file bulan baru, gabung dengan semua data lama, lalu langsung latih ulang."); b.clicked.connect(lambda: self.monthly()); row.addWidget(b)
        mb = QToolButton(); mb.setText("Lainnya ▾"); mb.setPopupMode(QToolButton.InstantPopup); m = QMenu(mb)
        for t, f in (("Impor folder…", lambda: self.upload(True)), ("Tempel teks JSON…", lambda: self.paste()), ("Unduh template format", lambda: self.template()),
                     ("Muat data contoh (uji coba saja)", lambda: self.demo()), ("Kosongkan semua data", lambda: self.clear())): m.addAction(t, f)
        mb.setMenu(m); row.addWidget(mb); row.addStretch()
        self.imp = QLabel(); self.imp.setObjectName("hint"); self.imp.setWordWrap(True); v1.addWidget(self.imp)
        # ② LATIH
        c2, v2 = card("②", "Latih model"); v.addWidget(c2); sr = QHBoxLayout(); self.cfg_sum = QLabel(); self.cfg_sum.setObjectName("hint"); sr.addWidget(self.cfg_sum, 1)
        self.advb = QToolButton(); self.advb.setText("Ubah pengaturan ▾"); self.advb.setCheckable(True); self.advb.toggled.connect(lambda on: self.adv.setVisible(on)); sr.addWidget(self.advb); v2.addLayout(sr)
        self.adv = QFrame(); self.adv.setObjectName("tile"); f = QFormLayout(self.adv); f.setContentsMargins(12, 10, 12, 10)
        self.mode = QComboBox(); self.mode.addItems(["Dari awal pada SEMUA data lama + baru (disarankan)", "Lanjutkan dari model aktif + data baru (warm start)"]); f.addRow("Mode latih", self.mode)
        self.target = QComboBox(); self.target.addItems(["Skor potensi order (0–100)", "Jumlah Order"]); f.addRow("Target", self.target)
        f.addRow("Catatan", QLabel("Skor membandingkan area; bukan peluang absolut mendapat order."))
        self.algo = QComboBox(); self.algo.addItems(["Otomatis: pilih terbaik (disarankan)", "MLP Neural Net (28-16-1)", "GLM L2 (regresi cepat)"]); f.addRow("Algoritma", self.algo)
        self.grid = QComboBox(); self.grid.addItems(["1.0", "0.5", "2.0"]); f.addRow("Ukuran grid (km)", self.grid)
        self.ep = QSpinBox(); self.ep.setRange(5, 500); self.ep.setValue(60); f.addRow("Epoch maksimum", self.ep)
        self.lm = QSpinBox(); self.lm.setRange(0, 60); f.addRow("Pakai N bulan terakhir (0 = semua)", self.lm)
        self.cv = QCheckBox("Validasi silang (lebih lama, hasil lebih bisa dipercaya)"); self.cv.setChecked(True); f.addRow(self.cv)
        self.cal = QCheckBox("Kalender libur Jawa Timur"); self.cal.setChecked(True); self.pay = QCheckBox("Tanggal gajian"); self.pay.setChecked(True); f.addRow(self.cal); f.addRow(self.pay)
        self.adv.hide(); v2.addWidget(self.adv); br = QHBoxLayout(); v2.addLayout(br)
        self.btn = QPushButton("Mulai Latih Model ONNX"); self.btn.setObjectName("primary"); self.btn.clicked.connect(lambda: self.start()); br.addWidget(self.btn)
        self.cancel = QPushButton("Batalkan"); self.cancel.hide(); self.cancel.clicked.connect(lambda: self.do_cancel()); br.addWidget(self.cancel); br.addStretch()
        self.bar = QProgressBar(); self.status = QLabel(); self.status.setObjectName("hint"); self.status.setWordWrap(True); v2.addWidget(self.bar); v2.addWidget(self.status)
        # ③ HASIL
        c3, v3 = card("③", "Hasil, uji & bagikan"); v.addWidget(c3)
        self.empty = QLabel("Belum ada model aktif. Selesaikan langkah ① dan ② dulu."); self.empty.setObjectName("muted"); v3.addWidget(self.empty)
        self.resf = QWidget(); rv = QVBoxLayout(self.resf); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(10); v3.addWidget(self.resf); self.resf.hide()
        top = QHBoxLayout(); self.qchip = QLabel(); self.mname = QLabel(); self.mname.setObjectName("title"); top.addWidget(self.qchip); top.addWidget(self.mname, 1); rv.addLayout(top)
        self.qtext = QLabel(); self.qtext.setWordWrap(True); self.qtext.setObjectName("muted"); rv.addWidget(self.qtext)
        kr = QHBoxLayout(); self.k1, self.k2, self.k3 = tile("Top-5"), tile("ROC-AUC"), tile("cakupan")
        for t in (self.k1, self.k2, self.k3): kr.addWidget(t[0])
        rv.addLayout(kr); self.info = QLabel(); self.info.setObjectName("hint"); self.info.setWordWrap(True); rv.addWidget(self.info)
        self.detb = QToolButton(); self.detb.setText("Lihat semua metrik ▾"); self.detb.setCheckable(True); self.detb.toggled.connect(lambda on: self.detail.setVisible(on)); rv.addWidget(self.detb, 0, Qt.AlignLeft)
        self.detail = QLabel(); self.detail.setObjectName("metrics"); self.detail.hide(); rv.addWidget(self.detail)
        ar = QHBoxLayout(); rv.addLayout(ar)
        self.go = QPushButton("⚡ Coba di Peta Radar"); self.go.setObjectName("primary"); self.go.clicked.connect(self.go_radar.emit); ar.addWidget(self.go)
        self.bt = QPushButton("🧪 Uji Paket"); self.bt.setToolTip("Menjalankan paket secara mandiri & membandingkannya dengan model di aplikasi"); self.bt.clicked.connect(lambda: self.test_pkg()); ar.addWidget(self.bt)
        self.be = QPushButton("📦 Ekspor Paket (.zip)"); self.be.setToolTip("Uji otomatis dulu; diekspor hanya jika lolos"); self.be.clicked.connect(lambda: self.test_pkg(True)); ar.addWidget(self.be)
        self.dl = QToolButton(); self.dl.setText("Unduh ▾"); self.dl.setPopupMode(QToolButton.InstantPopup); dm = QMenu(self.dl)
        dm.addAction("Model (.onnx)", lambda: self.save_file("onnx")); dm.addAction("Metadata (.json)", lambda: self.save_file("meta")); self.dl.setMenu(dm); ar.addWidget(self.dl); ar.addStretch()
        self.tbar = QProgressBar(); self.tbar.hide(); self.testres = QLabel(); self.testres.setObjectName("metrics"); self.testres.setWordWrap(True); rv.addWidget(self.tbar); rv.addWidget(self.testres)
        pv = QLabel("Catatan berbagi: _metadata.json berisi ringkasan jumlah order per area (bukan order per kejadian). Jangan dibagikan jika datanya sensitif."); pv.setObjectName("hint"); pv.setWordWrap(True); rv.addWidget(pv)
        v.addStretch(); self._pk_enable(False); self._load_settings(); self._sum(); self.refresh_report()

    # --- pengaturan tersimpan
    def _load_settings(self):
        q = self.qs
        for c, k in ((self.mode, "mode"), (self.target, "target"), (self.algo, "algo"), (self.grid, "grid")): c.setCurrentIndex(q.value(k, 0, type=int))
        self.ep.setValue(q.value("ep", 60, type=int)); self.lm.setValue(q.value("lm", 0, type=int)); self.cv.setChecked(q.value("cv", True, type=bool)); self.cal.setChecked(q.value("cal", True, type=bool)); self.pay.setChecked(q.value("pay", True, type=bool))
        for c in (self.mode, self.target, self.algo, self.grid): c.currentIndexChanged.connect(lambda *_: self._sum())
        self.ep.valueChanged.connect(lambda *_: self._sum())

    def _save_settings(self):
        q = self.qs
        for c, k in ((self.mode, "mode"), (self.target, "target"), (self.algo, "algo"), (self.grid, "grid")): q.setValue(k, c.currentIndex())
        q.setValue("ep", self.ep.value()); q.setValue("lm", self.lm.value()); q.setValue("cv", self.cv.isChecked()); q.setValue("cal", self.cal.isChecked()); q.setValue("pay", self.pay.isChecked())

    def _sum(self):
        self.cfg_sum.setText(f"Pengaturan: {['Otomatis', 'MLP', 'GLM'][self.algo.currentIndex()]} · {'skor potensi' if self.target.currentIndex() == 0 else 'jumlah'} · grid {self.grid.currentText()} km · "
                             f"{'dari awal' if self.mode.currentIndex() == 0 else 'lanjutan'} · maks {self.ep.value()} epoch")

    # --- data
    def refresh_report(self, files=0, added=0, dup=0, bad=()):
        r = fe.data_report(self.store); by = Counter(str(o.get("jenisOrder", "")) for o in self.store)
        self.t_n[1].setText(str(r["n"])); self.t_d[1].setText(str(r["days"])); self.t_a[1].setText(str(r["areas"]))
        chip(self.chip_ready, "good" if r["ready"] else "bad", "✔ Siap dilatih" if r["ready"] else "Belum cukup data")
        self.tips.setText("\n".join(r["tips"]) + (f"\nJenis: {', '.join(f'{k} {n}' for k, n in by.most_common())}" if by else ""))
        if files or added or dup or bad:
            self.imp.setText(f"Impor terakhir: {files} file · +{added} baru · {dup} duplikat · ✖ {len(bad)} ditolak" + "".join(f"\n  ✖ {k}: {n}x" for k, n in Counter(bad).most_common(4)))
        self.btn.setEnabled(r["ready"] and not self.training)
        if not self.training and not self.status.text().startswith(("✔", "✖", "Pelatihan")): self.status.setText("" if r["ready"] else "Siapkan data dulu di langkah ①.")

    def _merged(self, files, added, dup, bad):
        self.refresh_report(files, added, dup, bad); self.orders_changed.emit(); self.notify.emit(f"✔ +{added} order baru · {dup} duplikat · {len(bad)} ditolak")

    def upload(self, folder=False):
        r = import_json_dialog(self, self.store, folder is True)
        if r: self._merged(*r)

    def monthly(self):
        r = import_json_dialog(self, self.store)
        if not r: return
        self._merged(*r)
        if r[1] > 0: self.start()
        else: self.status.setText("Tidak ada order baru (semua duplikat/ditolak). Pelatihan dilewati.")

    def paste(self):
        t, ok = QInputDialog.getMultiLineText(self, "Tempel JSON", "Tempel teks JSON (daftar order):")
        if not ok or not t.strip(): return
        try: d = json.loads(t); self._merged(1, *fe.merge_orders(self.store, d if isinstance(d, list) else [d]))
        except Exception as e: QMessageBox.warning(self, "JSON tidak valid", str(e))

    def template(self):
        f, _ = QFileDialog.getSaveFileName(self, "Simpan template", "template_order.json", "JSON (*.json)")
        if f: json.dump(fe.template_orders(), open(f, "w", encoding="utf-8"), indent=1)

    def demo(self): self._merged(1, *fe.merge_orders(self.store, fe.demo_orders()))

    def clear(self):
        if QMessageBox.question(self, "Kosongkan", "Hapus SEMUA order tersimpan?") == QMessageBox.Yes:
            self.store.clear(); self.refresh_report(); self.orders_changed.emit()

    # --- latih
    def _busy(self, on):
        self.training = on; self.btn.setVisible(not on); self.cancel.setVisible(on)
        if on: self.bar.setValue(0)

    def start(self):
        if self.training: return
        if not fe.data_report(self.store)["ready"]: return self.status.setText("✖ Data belum cukup: butuh ≥ 100 order valid dan ≥ 7 hari berbeda.")
        self._save_settings()
        cfg = dict(target="prob" if self.target.currentIndex() == 0 else "count", algo=["auto", "mlp", "glm"][self.algo.currentIndex()], cv=self.cv.isChecked(), km=float(self.grid.currentText()),
                   epochs=self.ep.value(), calendar=self.cal.isChecked(), payday=self.pay.isChecked(), last_months=self.lm.value(), parent=Path(self.active).stem if self.active else None,
                   init_from=self.active if (self.mode.currentIndex() == 1 and self.active) else None)
        self._busy(True); self.w = Worker(list(self.store), cfg)
        self.w.prog.connect(lambda p, m: (self.bar.setValue(p), self.status.setText(m))); self.w.done.connect(self.finished); self.w.fail.connect(self.failed); self.w.start()

    def do_cancel(self):
        if self.training: self.w.stop = True; self.status.setText("Membatalkan…")

    def failed(self, e): self._busy(False); self.status.setText(("✖ " if "batal" not in e else "") + e); self.refresh_report()

    def finished(self, r):
        self._busy(False); self.result = r; self.status.setText("✔ Pelatihan selesai."); self.bar.setValue(100); self.show_model(r["meta"]); self._pk_enable(True)
        self.notify.emit(f"✔ Model {r['meta']['name']} selesai dilatih"); self.model_ready.emit(r["onnx_path"]); self.refresh_report()

    # --- hasil
    def show_model(self, m):
        x = m["metrics"]; q, qt = tr.quality_label(x); c, t = m.get("cleaning", {}), m.get("training", {})
        chip(self.qchip, {"Bagus": "good", "Cukup": "ok", "Setara": "ok"}.get(q, "bad"), f"Kualitas: {q}"); self.qtext.setText(qt)
        self.mname.setText(m["name"] + (f"   ← lanjutan dari {m['parent']}" if t.get("warm_start") and m.get("parent") else ""))
        self.k1[1].setText(f"{x.get('top5_hit_rate', 0):.0%}"); self.k1[2].setText(f"Top-5 model · tebakan sederhana terbaik {x.get('top5_baseline_best', x.get('top5_baseline_popular', 0)):.0%}" + (f" · selisih {x['lift_vs_best_baseline']:+.0%} (90%: {x['lift_ci_low']:+.0%} s/d {x['lift_ci_high']:+.0%})" if "lift_ci_low" in x else ""))
        self.k2[1].setText(f"{x.get('roc_auc', 0):.2f}"); self.k2[2].setText("ROC-AUC (0.5 = acak, 1 = sempurna)")
        self.k3[1].setText(f"{x.get('coverage', 0):.0%}"); self.k3[2].setText("order uji berada di area yang dikenal model")
        rem = "".join(f"\n  - dibuang ({k}): {n}" for k, n in c.get("removed", {}).items())
        self.info.setText(f"Data: {c.get('kept', '?')} order bersih dari {c.get('input', '?')} masuk{rem}\nSplit per hari: {t.get('days_train', '?')} latih / {t.get('days_val', '?')} validasi / {t.get('days_test', '?')} uji · "
                          f"epoch terbaik {t.get('best_epoch', '?')} dari {t.get('epochs_run', '?')}" + (" · LANJUTAN dari model aktif" if t.get("warm_start") else " · dilatih dari awal"))
        if t.get("candidates"): self.info.setText(self.info.text() + f"\nValidasi silang {t.get('cv_folds', 1)} lipatan · terpilih: {t['selected']} (aturan 1-SE) · " + " | ".join(f"{k} {v:.0%}" for k, v in t["candidates"].items()))
        self.detail.setText("\n".join(f"{k:<24}{v:>10.4f}" for k, v in x.items())); self.empty.hide(); self.resf.show()

    def _pk_enable(self, on):
        for b in (self.go, self.bt, self.be, self.dl): b.setEnabled(on)

    def set_active(self, path):  # model aktif di Peta Radar = model yang tampil/diuji/diekspor di sini
        self.active = path
        try: self.show_model(json.loads(Path(path).with_name(Path(path).stem + "_metadata.json").read_text("utf-8")))
        except Exception: pass
        self._pk_enable(True)

    def save_file(self, kind):
        if not self.active: return
        p = Path(self.active); src = p if kind == "onnx" else p.with_name(p.stem + "_metadata.json")
        f, _ = QFileDialog.getSaveFileName(self, "Simpan", src.name)
        if f: shutil.copy(src, f)

    # --- uji & ekspor paket (thread)
    def test_pkg(self, then_export=False):
        if not self.active: return
        self._then = then_export; self.tbar.setRange(0, 0); self.tbar.show(); self.testres.setText("Menguji paket di proses terpisah…")
        self._pk_enable(False); self.pw = PkgWorker(self.active); self.pw.done.connect(self._pkg_done); self.pw.start()

    def _pkg_done(self, r):
        self.tbar.hide(); self._pk_enable(True); self.testres.setText(r["report"])
        if not self._then: return
        if not r["ok"]: return QMessageBox.warning(self, "Uji gagal", "Paket TIDAK diekspor karena ada uji yang gagal. Lihat laporan di bawah tombol.")
        f, _ = QFileDialog.getSaveFileName(self, "Simpan paket", "driver_tracker_model.zip", "ZIP (*.zip)")
        if f: pk.export_zip(self.active, f); self.testres.setText(r["report"] + f"\n\n✔ Paket disimpan: {f}"); self.notify.emit("✔ Paket diekspor")
