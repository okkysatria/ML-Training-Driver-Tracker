import json, shutil, datetime as dt, heapq, math
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QUrl, QDateTime, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QToolButton, QMenu, QCheckBox, QInputDialog, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QComboBox, QListWidget,
                               QListWidgetItem, QTabBar, QTabWidget, QDateTimeEdit, QButtonGroup, QFileDialog, QMessageBox, QSplitter)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
import feature_engine as fe
import ml_onnx_trainer as tr

HTML = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}.pin{position:relative;width:40px;height:40px}
.ring{position:absolute;left:50%;top:50%;width:14px;height:14px;margin:-7px;border-radius:50%;border:2px solid var(--c);animation:pulse 2s infinite;opacity:0}
.r2{animation-delay:1s}@keyframes pulse{0%{transform:scale(1);opacity:.8}100%{transform:scale(3.2);opacity:0}}
.dot{position:absolute;left:50%;top:50%;width:16px;height:16px;margin:-8px;border-radius:50%;background:var(--d);border:2px solid #fff;box-shadow:0 1px 4px #0006;color:#fff;font:700 9px sans-serif;text-align:center;line-height:12px}
.driver-badge{position:absolute;left:50%;top:50%;width:30px;height:30px;margin:-15px;border-radius:50%;display:grid;place-items:center;background:#fff;border:2px solid #2563EB;box-shadow:0 1px 5px #0005}
.driver-badge svg{width:21px;height:21px;display:block}
.float .dot{animation:fl 1.4s ease-in-out infinite}@keyframes fl{50%{transform:translateY(-6px)}}
#lg{position:absolute;z-index:1000;right:10px;bottom:24px;background:#fff;padding:8px 10px;border-radius:8px;font:12px sans-serif;box-shadow:0 1px 6px #0003;line-height:1.7}
#lg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}</style></head>
<body><div id="m"></div>
<div id="lg"><b>Jenis order</b><br><i style="background:#16A34A"></i>Ride<br><i style="background:#EF4444"></i>Food<br><i style="background:#2563EB"></i>Paket/Send<br>
<svg style="width:15px;height:15px;vertical-align:-3px;margin-right:4px" viewBox="0 0 24 24" aria-label="Driver"><path d="M5 16h1l2-6h6l4 4h2a2 2 0 0 1 2 2v1h-2a2 2 0 0 1-4 0h-5a2 2 0 0 1-4 0H5v-1Zm4-4-1 3h8l-3-3H9Zm1-5h4l2 2h-7l1-2Z" fill="#2563EB"/><circle cx="8" cy="17" r="1" fill="#fff"/><circle cx="18" cy="17" r="1" fill="#fff"/></svg>Posisi driver<br><b>Cincin = skor potensi</b><br><i style="background:#059669"></i>&gt;45/100 <i style="background:#D97706"></i>25–45/100 <i style="background:#64748B"></i>&lt;25/100</div>
<script>
var map=L.map('m').setView([-7.2819,112.7381],12),pin=false;
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);
var A=L.layerGroup().addTo(map),B=L.layerGroup().addTo(map),C=L.layerGroup().addTo(map);
function ico(c,d,cl,t){return L.divIcon({className:'',iconSize:[40,40],iconAnchor:[20,20],html:'<div class="pin '+cl+'" style="--c:'+c+';--d:'+d+'"><div class="ring"></div><div class="ring r2"></div><div class="dot">'+t+'</div></div>'});}
function driverIco(){return L.divIcon({className:'',iconSize:[40,40],iconAnchor:[20,20],html:'<div class="pin" style="--c:#2563EB"><div class="ring"></div><div class="ring r2"></div><div class="driver-badge"><svg viewBox="0 0 24 24" aria-label="Driver"><path d="M5 16h1l2-6h6l4 4h2a2 2 0 0 1 2 2v1h-2a2 2 0 0 1-4 0h-5a2 2 0 0 1-4 0H5v-1Zm4-4-1 3h8l-3-3H9Zm1-5h4l2 2h-7l1-2Z" fill="#2563EB"/><circle cx="8" cy="17" r="1" fill="#fff"/><circle cx="18" cy="17" r="1" fill="#fff"/></svg></div></div>'});}
function setDriver(la,lo,r,fit){A.clearLayers();L.marker([la,lo],{icon:driverIco()}).bindPopup('Posisi driver').addTo(A);
L.circle([la,lo],{radius:r*1000,color:'#2563EB',weight:1,fillOpacity:.04}).addTo(A);if(fit)map.setView([la,lo],r<=2?14:r<=3.5?13:12);}
function setSpots(a){B.clearLayers();a.forEach(function(s,i){L.marker([s.lat,s.lon],{icon:ico(s.c,s.d,s.p>.45?'float':'',i<9?i+1:'')}).bindPopup('<b>'+s.n+'</b><br>Skor potensi '+Math.round(s.p*100)+'/100<br>Dominan: '+s.j).addTo(B);});}
function setHist(a){C.clearLayers();a.forEach(function(h){L.circleMarker([h.lat,h.lon],{radius:5,color:'#fff',weight:1,fillColor:h.c,fillOpacity:.9}).bindPopup(h.t).addTo(C);});}
function showHistory(b){if(b){if(!map.hasLayer(C))C.addTo(map);}else if(map.hasLayer(C)){map.removeLayer(C);}}
function focusSpot(la,lo){map.flyTo([la,lo],16);}
function setPinMode(b){pin=b;map.getContainer().style.cursor=b?'crosshair':'';}
map.on('click',function(e){if(pin)console.log('PIN:'+e.latlng.lat.toFixed(6)+','+e.latlng.lng.toFixed(6));});
</script></body></html>"""


class Page(QWebEnginePage):
    pinned = Signal(float, float)

    def javaScriptConsoleMessage(self, level, msg, line, src):  # saluran JS -> Python untuk klik pin
        if msg.startswith("PIN:"):
            a, b = msg[4:].split(","); self.pinned.emit(float(a), float(b))


class PagedList(QWidget):
    clicked = Signal(object)

    def __init__(self, size=7):
        super().__init__(); self.size, self.items, self.page = size, [], 0
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); self.lw = QListWidget(); v.addWidget(self.lw)
        h = QHBoxLayout(); self.pv, self.nx, self.lb = QPushButton("‹"), QPushButton("›"), QLabel(); self.lb.setObjectName("mono")
        for w in (self.pv, self.lb, self.nx): h.addWidget(w)
        v.addLayout(h); self.pv.clicked.connect(lambda: self._go(-1)); self.nx.clicked.connect(lambda: self._go(1))
        self.lw.itemClicked.connect(lambda it: self.clicked.emit(it.data(Qt.UserRole)))

    def set_items(self, items): self.items, self.page = items, 0; self._show()

    def _go(self, d): self.page = max(0, min(self.page + d, (len(self.items) - 1) // self.size)); self._show()

    def _show(self):
        self.lw.clear()
        for t, p, *c in self.items[self.page * self.size:(self.page + 1) * self.size]:
            it = QListWidgetItem(t); it.setData(Qt.UserRole, p)
            if c: it.setForeground(QColor(c[0]))
            self.lw.addItem(it)
        self.lb.setText(f"{self.page + 1}/{max(1, -(-len(self.items) // self.size))}  ({len(self.items)})")


class RadarView(QWidget):
    train_requested = Signal()
    order_logged = Signal()
    model_loaded = Signal(str)
    notify = Signal(str)

    def __init__(self, orders):
        super().__init__(); self.setObjectName("page")
        self.orders, self.rt, self.ready, self.sim, self.history_visible = orders, None, False, False, True
        self.names = list(fe.HOTSPOTS); self.offsets = [0, 15]; self.jenis = "Semua"; self.pos = fe.HOTSPOTS[self.names[0]][:2]
        root = QVBoxLayout(self); root.setContentsMargins(16, 12, 16, 12); root.setSpacing(10)
        hd = QFrame(); hd.setObjectName("card"); h = QHBoxLayout(hd); h.setContentsMargins(16, 10, 16, 10)
        self.lbl = QLabel("Belum ada model"); self.lbl.setObjectName("title"); self.sub = QLabel("Buat dulu: menu Model ▾ → Latih model baru"); self.sub.setObjectName("hint")
        col = QVBoxLayout(); col.setSpacing(1); col.addWidget(self.lbl); col.addWidget(self.sub); h.addLayout(col, 1)
        self.qchip = QLabel(); self.qchip.hide(); h.addWidget(self.qchip)
        mb = QToolButton(); mb.setText("Model ▾"); mb.setPopupMode(QToolButton.InstantPopup); mn = QMenu(mb)
        mn.addAction("Latih model baru…", lambda: self.train_requested.emit()); mn.addAction("Impor ONNX…", lambda: self.import_onnx()); mn.addAction("Muat model terakhir", lambda: self.load_latest())
        mb.setMenu(mn); h.addWidget(mb); root.addWidget(hd)
        bar = QFrame(); bar.setObjectName("card"); bv = QVBoxLayout(bar); bv.setContentsMargins(16, 10, 16, 10); bv.setSpacing(8)
        r1 = QHBoxLayout(); r1.setSpacing(8); bv.addLayout(r1); r1.addWidget(QLabel("📍 Posisi"))
        self.loc = QComboBox(); self.loc.addItems(self.names); self.loc.setToolTip("Pilih hotspot sebagai posisi Anda"); r1.addWidget(self.loc)
        self.pinbtn = QPushButton("Pin di peta"); self.pinbtn.setCheckable(True); self.pinbtn.setToolTip("Nyalakan, lalu klik titik di peta untuk menaruh posisi Anda"); self.pinbtn.toggled.connect(self.pin_mode); r1.addWidget(self.pinbtn)
        self.coord = QLabel(); self.coord.setObjectName("mono"); r1.addWidget(self.coord); r1.addSpacing(10); r1.addWidget(QLabel("Jenis")); self.jg = QButtonGroup(self)
        for t in ("Semua", "Food", "Ride", "Send"):
            b = QPushButton(fe.JENIS_LABEL.get(t, t)); b.setProperty("jenis_key", t); b.setCheckable(True); b.setChecked(t == "Semua")
            b.setToolTip("Tampilkan hanya area yang punya riwayat jenis ini" if t != "Semua" else "Semua jenis order")
            if t in fe.JENIS_COLOR: b.setStyleSheet(f"QPushButton{{border-left:4px solid {fe.JENIS_COLOR[t]}}}")
            self.jg.addButton(b); r1.addWidget(b)
        self.jg.buttonClicked.connect(lambda b: (setattr(self, "jenis", b.property("jenis_key")), self.refresh()))
        r1.addSpacing(10); r1.addWidget(QLabel("Radius")); self.rad = QComboBox(); self.rad.addItems(["2 km", "3.5 km", "5 km"]); self.rad.setCurrentIndex(1); r1.addWidget(self.rad); r1.addStretch()
        r2 = QHBoxLayout(); r2.setSpacing(8); bv.addLayout(r2); r2.addWidget(QLabel("⏱ Rentang")); self.tg = QButtonGroup(self)
        self.ranges = {"0–15m": [0, 15], "+15m": [15], "+30m": [30], "+45m": [45], "+60m": [60], "Semua": [0, 15, 30, 45, 60]}
        for t in self.ranges:
            b = QPushButton(t); b.setCheckable(True); b.setChecked(t == "0–15m"); self.tg.addButton(b); r2.addWidget(b)
        self.tg.buttonClicked.connect(lambda b: (setattr(self, "offsets", self.ranges[b.text()]), self.refresh()))
        r2.addSpacing(10); self.simchk = QCheckBox("Simulasi hari dan jam"); self.simchk.setToolTip("Matikan = pakai waktu sekarang (otomatis diperbarui tiap menit). Nyalakan untuk menguji hari/jam lain."); self.simchk.toggled.connect(self.toggle_sim); r2.addWidget(self.simchk)
        self.histchk = QCheckBox("Tampilkan riwayat"); self.histchk.setChecked(True); self.histchk.setToolTip("Sembunyikan/tampilkan titik riwayat di peta dan panel riwayat."); self.histchk.toggled.connect(self.toggle_history); r2.addWidget(self.histchk)
        self.badge = QLabel(); self.badge.setStyleSheet("color:#4F46E5;font-weight:600"); r2.addWidget(self.badge, 1)
        lb = QPushButton("+ Catat Order di Pin"); lb.setObjectName("dark"); lb.setToolTip("Simpan order baru (jenis, jam sekarang, posisi pin) sebagai data latih"); lb.clicked.connect(lambda: self.log_order()); r2.addWidget(lb)
        self.simpanel = QWidget(); r3 = QHBoxLayout(self.simpanel); r3.setContentsMargins(0, 0, 0, 0); self.tabs = QTabBar()
        for d in fe.DAYS: self.tabs.addTab(d.capitalize())
        r3.addWidget(self.tabs); self.dte = QDateTimeEdit(QDateTime.currentDateTime()); self.dte.setCalendarPopup(True); self.dte.setDisplayFormat("dd/MM/yyyy HH:mm"); r3.addWidget(self.dte)
        nb = QPushButton("Sekarang"); nb.clicked.connect(lambda: self.dte.setDateTime(QDateTime.currentDateTime())); r3.addWidget(nb); r3.addStretch(); self.simpanel.hide(); bv.addWidget(self.simpanel); root.addWidget(bar)
        self.tip = QLabel(); self.tip.setWordWrap(True); root.addWidget(self.tip)
        sp = QSplitter(); self.web = QWebEngineView(); self.pg = Page(self.web); self.web.setPage(self.pg)
        self.pg.pinned.connect(self.pinned); self.web.loadFinished.connect(self._loaded); self.web.setHtml(HTML, QUrl("https://localhost/")); sp.addWidget(self.web)
        self.side = QTabWidget(); self.spots, self.hist = PagedList(), PagedList()
        self.side.addTab(self.spots, "Titik Rekomendasi (ML)"); self.side.addTab(self.hist, "Riwayat Order Terdekat")
        self.spots.clicked.connect(self.focus); self.hist.clicked.connect(self.focus); sp.addWidget(self.side); self.side.setMinimumWidth(380); sp.setChildrenCollapsible(False); sp.setStretchFactor(0, 3); sp.setStretchFactor(1, 2); sp.setSizes([800, 420]); root.addWidget(sp, 1)
        self.tabs.setCurrentIndex(dt.date.today().weekday()); self.loc.currentIndexChanged.connect(self.pick_hotspot); self.rad.currentIndexChanged.connect(self.relocate)
        self.tabs.currentChanged.connect(self.day_changed); self.dte.dateTimeChanged.connect(self.time_changed)
        self.clock = QTimer(self); self.clock.timeout.connect(self.tick); self.clock.start(60000); self._coord(); self.time_changed()

    def _tip(self, text, kind="ok"):
        c = {"ok": "background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0", "warn": "background:#FFFBEB;color:#92400E;border:1px solid #FDE68A"}[kind]
        self.tip.setStyleSheet(f"{c};border-radius:10px;padding:10px;font-weight:600"); self.tip.setText(text)

    def when(self): return self.dte.dateTime().toPython() if self.sim else dt.datetime.now()
    def radius(self): return float(self.rad.currentText().split()[0])
    def js(self, s): self.web.page().runJavaScript(s)
    def _coord(self): self.coord.setText(f"{self.pos[0]:.5f}, {self.pos[1]:.5f}")
    def _loaded(self, ok):
        self.ready = ok
        if ok: self.js(f"showHistory({'true' if self.history_visible else 'false'})")
        self.relocate()
    def pin_mode(self, on): self.js(f"setPinMode({'true' if on else 'false'})")

    def pinned(self, la, lo):  # klik peta saat mode Pin aktif -> posisi driver
        self.pos = (la, lo); self._coord(); self.js(f"setDriver({la},{lo},{self.radius()},false)"); self.refresh()

    def pick_hotspot(self): self.pos = fe.HOTSPOTS[self.loc.currentText()][:2]; self._coord(); self.relocate()

    def relocate(self):
        if self.ready: self.js(f"setDriver({self.pos[0]},{self.pos[1]},{self.radius()},true)"); self.refresh()

    def toggle_sim(self, on):
        self.sim = on; self.simpanel.setVisible(on)
        if on: self.dte.setDateTime(QDateTime.currentDateTime())
        self.time_changed()

    def day_changed(self, i):
        d = self.dte.dateTime().date(); delta = (i - (d.dayOfWeek() - 1)) % 7
        self.dte.blockSignals(True); self.dte.setDate(d.addDays(delta)); self.dte.blockSignals(False); self.time_changed()

    def time_changed(self):
        w = self.when(); self.tabs.blockSignals(True); self.tabs.setCurrentIndex(w.weekday()); self.tabs.blockSignals(False)
        self.badge.setText(f"{fe.DAYS[w.weekday()].capitalize()} {w:%H:%M} " + ("(simulasi) · " if self.sim else "(sekarang) · ") + " · ".join(fe.flags(w, self.rt.meta["opts"] if self.rt else None))); self.refresh()

    def tick(self):
        if not self.sim: self.time_changed()

    def toggle_history(self, on):
        self.history_visible = on
        self.side.setTabVisible(1, on)
        if self.ready: self.js(f"showHistory({'true' if on else 'false'})")
        self.refresh()

    def focus(self, p):
        if p: self.js(f"focusSpot({p[0]},{p[1]})")

    def log_order(self):
        j, ok = QInputDialog.getItem(self, "Catat Order", "Jenis order yang baru kamu dapat:", ["Ride", "Food", "Paket"], 0, False)
        if not ok: return
        n = dt.datetime.now(); la, lo = self.pos  # titik akhir belum diketahui -> diisi sama dengan titik jemput
        o = dict(latitudePickup=la, longitudePickup=lo, latitudeAkhir=la, longitudeAkhir=lo, jamMulai=n.strftime("%H:%M"), tanggal=n.date().isoformat(), jenisOrder=j)
        if fe.merge_orders(self.orders, [o])[0]: self.order_logged.emit(); self.notify.emit(f"✔ Order {j} dicatat pada {n:%H:%M}")

    def load_model(self, path):
        self.rt = tr.Runtime(path); m = self.rt.meta; x = m["metrics"]; q = tr.quality_label(x)[0]; self.lbl.setText(f"Model aktif: {m['name']}")
        self.sub.setText(f"{m['algorithm']} · Top-5 {x['top5_hit_rate']:.0%} (tebakan sederhana {x.get('top5_baseline_best', x.get('top5_baseline_popular', 0)):.0%}) · {m.get('n_orders', '?')} order")
        self.qchip.setText(f"Kualitas: {q}"); self.qchip.setProperty("chip", {"Bagus": "good", "Cukup": "ok", "Setara": "ok"}.get(q, "bad")); self.qchip.style().unpolish(self.qchip); self.qchip.style().polish(self.qchip); self.qchip.show()
        self.time_changed(); self.model_loaded.emit(str(path))

    def load_latest(self):
        p = tr.latest_model()
        if p: self.load_model(p)
        else: QMessageBox.information(self, "Model", "Belum ada model. Latih dulu di Studio Pelatihan.")

    def import_onnx(self):
        f, _ = QFileDialog.getOpenFileName(self, "Impor ONNX", "", "ONNX (*.onnx)")
        if not f: return
        p = Path(f); meta = p.with_name(p.stem + "_metadata.json")
        if not meta.exists(): return QMessageBox.warning(self, "Gagal", f"File {meta.name} harus berada satu folder dengan .onnx.")
        try:
            dst = tr.MODELS / p.name
            if p.resolve() != dst.resolve(): shutil.copy(p, dst); shutil.copy(meta, tr.MODELS / meta.name)
            self.load_model(str(dst))
        except Exception as e: QMessageBox.critical(self, "Gagal impor", str(e))

    def refresh(self):
        if not self.ready: return
        la, lo = self.pos; r = self.radius()
        def history_rows():
            for o in self.orders:
                try: a, b = float(o["latitudePickup"]), float(o["longitudePickup"])
                except Exception: continue
                d = fe.hav(la, lo, a, b)
                if d <= r and (self.jenis == "Semua" or str(o.get("jenisOrder", "")).capitalize() == self.jenis): yield d, a, b, o
        near = heapq.nsmallest(300, history_rows(), key=lambda x: x[0]) if self.history_visible else []
        near.sort(key=lambda x: x[0]); col = lambda o: fe.JENIS_COLOR.get(str(o.get("jenisOrder", "")).capitalize(), "#64748B")
        self.js("setHist(%s)" % json.dumps([dict(lat=a, lon=b, c=col(o), t=f"{o.get('tanggal')} {o.get('jamMulai')} · {fe.JENIS_LABEL.get(str(o.get('jenisOrder', '')).capitalize(), o.get('jenisOrder'))}") for _, a, b, o in near]))
        self.hist.set_items([(f"● {o.get('tanggal')} {o.get('jamMulai')} · {fe.JENIS_LABEL.get(str(o.get('jenisOrder', '')).capitalize(), o.get('jenisOrder'))} · {d:.1f} km", (a, b), col(o)) for d, a, b, o in near])
        if not self.rt: self.js("setSpots([])"); self.spots.set_items([]); self._tip("Belum ada model. Buka menu Model ▾ → Latih model baru… (atau halaman Mulai untuk panduan).", "warn"); return
        # Rekomendasi hanya dibuat pada koordinat pickup historis (dikelompokkan
        # rapat sekitar 10 m), bukan pusat sel grid yang bisa jatuh di lokasi kosong.
        m, km, points = self.rt.meta, self.rt.meta["grid_km"], {}
        for o in self.orders:
            rec, err = fe.parse_order(o, km)
            if err: continue
            jn = rec["jenis"]
            if self.jenis != "Semua" and jn != self.jenis: continue
            key = (round(rec["lat"], 4), round(rec["lon"], 4), jn)
            if key not in points: points[key] = [0.0, 0.0, []]
            points[key][0] += rec["lat"]; points[key][1] += rec["lon"]; points[key][2].append(rec)
        cells = []
        for (_, _, jn), (sum_la, sum_lo, history) in points.items():
            cl, co = sum_la / len(history), sum_lo / len(history)
            dist = fe.hav(la, lo, cl, co)
            if dist <= r: cells.append((fe.cell_of(cl, co, km), cl, co, jn, dist, history))
        # Batasi kandidat terdekat agar refresh tetap ringan pada riwayat besar.
        cells = sorted(cells, key=lambda c: c[4])[:1500]
        base = self.when(); targets = [base + dt.timedelta(minutes=o) for o in self.offsets]
        P = self.rt.predict([c[0] for c in cells], targets) if cells else []
        # Tambahkan bukti permintaan di titik ini pada slot/hari/bulan/libur serupa.
        # Dukungan makin kuat jika banyak order historis benar-benar tercatat dekat titik.
        scores = []
        for i, (_, cl, co, jn, _, history) in enumerate(cells):
            model_score = float(P[i].max())
            evidence = 0.0
            for target in targets:
                slot = fe.slot_of(target.hour + target.minute / 60)
                for rec in history:
                    old = rec["when"]
                    temporal = (1.0 if fe.slot_of(old.hour + old.minute / 60) == slot else .18)
                    temporal *= 1.0 if old.weekday() == target.weekday() else .35
                    temporal *= 1.0 if old.month == target.month else .60
                    temporal *= 1.0 if fe.is_holiday(old.date()) == fe.is_holiday(target.date()) else .60
                    age_days = max(0, (target.date() - old.date()).days)
                    temporal *= .5 ** (age_days / 365.0)
                    evidence += temporal / max(len(targets), 1)
            confidence = 1.0 - math.exp(-evidence / 3.0)
            scores.append(model_score * (.4 + .6 * confidence))
        order = sorted(range(len(cells)), key=lambda i: -scores[i])[:40]; items, pins = [], []
        for n, i in enumerate(order, 1):
            _, cl, co, jn, dist, _ = cells[i]; p = scores[i]; jc = fe.JENIS_COLOR.get(jn, "#64748B")
            name = fe.nearest_name(cl, co, *fe.cell_of(cl, co, km))
            if name.startswith("Grid "): name = "Titik riwayat"
            pins.append(dict(lat=cl, lon=co, p=p, n=name, j=fe.JENIS_LABEL.get(jn, jn), d=jc, c="#059669" if p > .45 else "#D97706" if p >= .25 else "#64748B"))
            items.append((f"{n}. {name} — skor {p * 100:.0f}/100 · {fe.JENIS_LABEL.get(jn, jn)} · {dist:.1f} km", (cl, co), jc))
        self.js(f"setSpots({json.dumps(pins)})"); self.spots.set_items(items)
        if pins: self._tip(f"🏆 Saran utama: {pins[0]['n']} — skor potensi {pins[0]['p'] * 100:.0f}/100 · {items[0][0].split(' · ')[-1]} · waktu {base:%H:%M}")
        else: self._tip("Belum ada titik pickup riwayat untuk radius/jenis ini. Perbesar radius atau pindahkan pin.", "warn")
