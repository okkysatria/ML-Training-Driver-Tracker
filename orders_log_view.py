from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSortFilterProxyModel, QDate, QTimer
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QTableView, QDialog, QFormLayout, QLabel, QMessageBox,
                               QDoubleSpinBox, QDateEdit, QComboBox, QDialogButtonBox, QHeaderView, QFileDialog)
import feature_engine as fe

COLS = ["tanggal", "jamMulai", "jenisOrder", "latitudePickup", "longitudePickup", "latitudeAkhir", "longitudeAkhir"]


def import_json_dialog(parent, store, folder=False):
    """Pilih BANYAK file JSON, gabungkan ke store. Return (n_file, baru, duplikat, errors) atau None."""
    if folder:  # semua .json/.csv dalam satu folder (mis. folder "Oktober 2026")
        d = QFileDialog.getExistingDirectory(parent, "Pilih folder berisi file JSON/CSV")
        files = sorted(str(p) for p in Path(d).glob("*") if p.suffix.lower() in (".json", ".csv")) if d else []
    else: files, _ = QFileDialog.getOpenFileNames(parent, "Pilih file JSON/CSV (gunakan Ctrl atau Shift untuk memilih banyak)", "", "Data order (*.json *.csv)")
    if not files: return None
    a = d = 0; bad = []
    for f in files:
        try:
            x, y, z = fe.merge_orders(store, fe.load_json_records(f)); a += x; d += y; bad += z
        except Exception as e:
            bad.append(f"{f.replace(chr(92), '/').split('/')[-1]}: {e}")
    return len(files), a, d, bad


class OrdersLogView(QWidget):
    changed = Signal()

    def __init__(self, orders):
        super().__init__(); self.setObjectName("page"); self.orders = orders
        v = QVBoxLayout(self); v.setContentsMargins(24, 18, 24, 18); v.setSpacing(10); h = QHBoxLayout(); v.addLayout(h)
        self.q = QLineEdit(); self.q.setPlaceholderText("Cari tanggal, jam, jenis…"); h.addWidget(self.q, 1)
        for t, f, o in (("Impor JSON…", self.imp, "dark"), ("+ Tambah Order", self.add, "primary"), ("Hapus Terpilih", self.delete, "")):
            b = QPushButton(t); b.setObjectName(o); b.clicked.connect(f); h.addWidget(b)
        self.info = QLabel(); self.info.setObjectName("muted"); v.addWidget(self.info)
        nav = QHBoxLayout(); v.addLayout(nav)
        self.prev = QPushButton("‹ Sebelumnya"); self.next = QPushButton("Berikutnya ›"); self.page_label = QLabel(); self.page_label.setObjectName("mono")
        self.page_size = QComboBox(); self.page_size.addItems(["100", "250", "500"]); self.page_size.setCurrentText("250")
        nav.addWidget(self.prev); nav.addWidget(self.page_label); nav.addWidget(self.next); nav.addStretch(); nav.addWidget(QLabel("Baris per halaman")); nav.addWidget(self.page_size)
        self.model = QStandardItemModel(0, len(COLS)); self.model.setHorizontalHeaderLabels(COLS)
        self.proxy = QSortFilterProxyModel(); self.proxy.setSourceModel(self.model); self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.tv = QTableView(); self.tv.setModel(self.proxy); self.tv.setSortingEnabled(True); self.tv.setAlternatingRowColors(True)
        self.tv.setSelectionBehavior(QTableView.SelectRows); self.tv.setEditTriggers(QTableView.NoEditTriggers)
        self.tv.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); v.addWidget(self.tv)
        self.filtered_indices = None; self.page = 0; self.search_timer = QTimer(self); self.search_timer.setSingleShot(True); self.search_timer.setInterval(180)
        self.search_timer.timeout.connect(self._apply_search); self.q.textChanged.connect(lambda *_: self.search_timer.start())
        self.prev.clicked.connect(lambda: self._change_page(-1)); self.next.clicked.connect(lambda: self._change_page(1))
        self.page_size.currentTextChanged.connect(self._page_size_changed)
        self.reload()

    def reload(self):
        """Render hanya satu halaman supaya riwayat besar tidak membuat tabel berat."""
        self.page = 0
        self._apply_search()

    def _apply_search(self):
        term = self.q.text().strip().casefold()
        self.filtered_indices = ([i for i, o in enumerate(self.orders)
                                  if term in " ".join(str(o.get(c, "")) for c in COLS).casefold()] if term else None)
        self._show_page()

    def _page_size_changed(self, text):
        self.page = 0; self._show_page()

    def _change_page(self, delta):
        self.page += delta; self._show_page()

    def _show_page(self):
        """Bangun maksimal page_size baris, bukan seluruh riwayat."""
        total = len(self.orders) if self.filtered_indices is None else len(self.filtered_indices)
        size = int(self.page_size.currentText()); pages = max(1, (total + size - 1) // size)
        self.page = max(0, min(self.page, pages - 1)); start = self.page * size; stop = min(start + size, total)
        indices = range(start, stop) if self.filtered_indices is None else self.filtered_indices[start:stop]
        self.tv.setSortingEnabled(False)
        self.model.removeRows(0, self.model.rowCount())
        for i in indices: self._row(self.orders[i], i)
        self.tv.setSortingEnabled(True)
        self.prev.setEnabled(self.page > 0); self.next.setEnabled(self.page + 1 < pages)
        self.page_label.setText(f"Halaman {self.page + 1}/{pages} · {total:,} order" if total else "Halaman 1/1 · 0 order")
        if self.q.text().strip(): self.info.setText(f"Menampilkan {total:,} hasil pencarian dari {len(self.orders):,} order tersimpan.")
        else: self.info.setText(f"{len(self.orders):,} order tersimpan (gabungan impor dan input manual).")

    def _row(self, o, original_index):
        it = [QStandardItem(str(o.get(c, ""))) for c in COLS]
        it[0].setData(original_index, Qt.UserRole)
        it[2].setForeground(QColor(fe.JENIS_COLOR.get(str(o.get("jenisOrder", "")), "#64748B"))); self.model.appendRow(it)

    def imp(self):
        r = import_json_dialog(self, self.orders)
        if r: self.reload(); self.changed.emit()

    def delete(self):
        rows = sorted({self.model.item(self.proxy.mapToSource(i).row(), 0).data(Qt.UserRole)
                       for i in self.tv.selectionModel().selectedRows()}, reverse=True)
        for r in rows: del self.orders[r]
        if rows: self.reload(); self.changed.emit()

    def add(self):
        d = QDialog(self); d.setWindowTitle("Tambah Order"); f = QFormLayout(d); sp = []
        for lbl, val in (("Lat Pickup", -7.2625), ("Lng Pickup", 112.7381), ("Lat Akhir", -7.2655), ("Lng Akhir", 112.7523)):
            s = QDoubleSpinBox(); s.setDecimals(6); s.setRange(-180, 180); s.setValue(val); f.addRow(lbl, s); sp.append(s)
        tm = QLineEdit("08:00"); dd = QDateEdit(QDate.currentDate()); jn = QComboBox(); jn.addItems(["Ride", "Food", "Paket"])
        f.addRow("Jam (HH:MM)", tm); f.addRow("Tanggal", dd); f.addRow("Jenis", jn)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(d.accept); bb.rejected.connect(d.reject); f.addRow(bb)
        if d.exec():
            o = dict(latitudePickup=sp[0].value(), longitudePickup=sp[1].value(), latitudeAkhir=sp[2].value(), longitudeAkhir=sp[3].value(),
                     jamMulai=tm.text().strip(), tanggal=dd.date().toString("yyyy-MM-dd"), jenisOrder=jn.currentText())
            added, duplicates, bad = fe.merge_orders(self.orders, [o])
            if added:
                self.reload(); self.changed.emit()
            elif bad:
                QMessageBox.warning(self, "Order tidak valid", bad[0])
            else:
                QMessageBox.information(self, "Order duplikat", "Order ini sudah ada di riwayat dan tidak ditambahkan.")
