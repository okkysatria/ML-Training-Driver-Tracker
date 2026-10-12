"""Bangun & uji 'paket model' mandiri (ONNX + metadata + kode prediksi) untuk dipakai aplikasi lain."""
import json, shutil, subprocess, sys, tempfile, zipfile, datetime as dt
from pathlib import Path
import numpy as np
import feature_engine as fe
import ml_onnx_trainer as tr

SCEN = [dt.datetime(2026, 10, 12, 8, 0), dt.datetime(2026, 10, 16, 18, 0), dt.datetime(2026, 10, 18, 3, 0), dt.datetime(2026, 10, 14, 12, 30)]
SCEN_NAMES = ["Sen 08:00", "Jum 18:00", "Min 03:00", "Rab 12:30"]

PREDICT_PY = r'''#!/usr/bin/env python3
"""Prediksi potensi order driver. Butuh: pip install onnxruntime numpy
Contoh CLI : python predict_demand.py --lat -7.2625 --lon 112.7381 --when "2026-10-16 18:00"
Contoh kode: from predict_demand import DemandPredictor; DemandPredictor().top_spots(-7.26, 112.74)"""
import argparse, json, datetime as dt
from pathlib import Path
import numpy as np
import onnxruntime as ort
import feature_engine as fe

HERE = Path(__file__).parent
DEFAULT_MODEL = "__MODEL__"
SCEN = [dt.datetime(2026, 10, 12, 8, 0), dt.datetime(2026, 10, 16, 18, 0), dt.datetime(2026, 10, 18, 3, 0), dt.datetime(2026, 10, 14, 12, 30)]


class DemandPredictor:
    def __init__(self, onnx_path=None):
        p = Path(onnx_path) if onnx_path else HERE / DEFAULT_MODEL
        self.meta = json.loads(p.with_name(p.stem + "_metadata.json").read_text("utf-8"))
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.mn = np.array(self.meta["scaler"]["min"], np.float32)
        self.mx = np.array(self.meta["scaler"]["max"], np.float32)
        self.km = self.meta["grid_km"]

    def predict_cells(self, cells, whens):
        """cells=[(gx,gy)], whens=[datetime] -> array peluang 0-1 berukuran (len(cells), len(whens))."""
        m, rows = self.meta, []
        for c in cells:
            k = f"{c[0]},{c[1]}"
            for w in whens:
                rows.append(fe.extract(c[0], c[1], w, fe.get_priors(m, c[0], c[1], w), m["opts"]))
        X = ((np.array(rows, np.float32) - self.mn) / (self.mx - self.mn + 1e-9)).astype(np.float32)
        o = self.sess.run(None, {"input": X})[0].ravel()
        p = o if m["target"] == "prob" else 1 - np.exp(-o)
        return np.clip(p, 0, 1).reshape(len(cells), len(whens))

    def top_spots(self, lat, lon, when=None, radius_km=3.5, n=5, horizon_min=(0, 15, 30, 45, 60), jenis=None):
        when = when or dt.datetime.now(); m, cells = self.meta, []
        for k in m["cell_rate"]:
            gx, gy = map(int, k.split(",")); cl, co = fe.cell_center(gx, gy, self.km); d = fe.hav(lat, lon, cl, co)
            ty = m["cell_types"].get(k, {})
            if d <= radius_km and (not jenis or ty.get(jenis)): cells.append((gx, gy, cl, co, d, ty))
        if not cells: return []
        P = self.predict_cells([c[:2] for c in cells], [when + dt.timedelta(minutes=x) for x in horizon_min]).max(axis=1)
        out = []
        for i in np.argsort(-P)[:n]:
            gx, gy, cl, co, d, ty = cells[i]
            out.append(dict(name=fe.nearest_name(cl, co, gx, gy), lat=round(cl, 6), lon=round(co, 6), probability=round(float(P[i]), 4),
                            distance_km=round(d, 2), dominant_jenis=max(ty, key=ty.get) if ty else None))
        return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--model"); a.add_argument("--lat", type=float); a.add_argument("--lon", type=float)
    a.add_argument("--when"); a.add_argument("--radius", type=float, default=3.5); a.add_argument("--n", type=int, default=5)
    a.add_argument("--jenis"); a.add_argument("--selftest", action="store_true"); g = a.parse_args(); mdl = DemandPredictor(g.model)
    if g.selftest:
        cells = [tuple(map(int, k.split(","))) for k in list(mdl.meta["cell_rate"])[:30]]
        print(json.dumps({"pred": mdl.predict_cells(cells, SCEN).tolist()}))
    elif g.lat is None or g.lon is None: a.error("isi --lat dan --lon (atau pakai --selftest)")
    else:
        w = dt.datetime.fromisoformat(g.when) if g.when else None
        print(json.dumps(mdl.top_spots(g.lat, g.lon, w, g.radius, g.n, jenis=g.jenis), ensure_ascii=False, indent=1))
'''

README = r'''# Paket Model Prediksi Order Driver

Isi paket:
- `__MODEL__` + `__STEM___metadata.json` — model ONNX dan konfigurasi (grid, 28 nama fitur, scaler, metrik).
- `feature_engine.py` — pembuat fitur (hanya library standar Python). WAJIB ikut dibawa.
- `predict_demand.py` — pembungkus siap pakai.

## Pasang
    pip install onnxruntime numpy

## Pakai di Python (aplikasi lain)
    from predict_demand import DemandPredictor
    import datetime as dt
    m = DemandPredictor()   # memuat __MODEL__ di folder ini
    spots = m.top_spots(lat=-7.2625, lon=112.7381, when=dt.datetime(2026, 10, 16, 18, 0), radius_km=3.5, n=5)
    # -> [{"name":"Tunjungan Plaza","lat":..,"lon":..,"probability":0.62,"distance_km":1.2,"dominant_jenis":"Food"}, ...]

## Pakai dari command line (output JSON)
    python predict_demand.py --lat -7.2625 --lon 112.7381 --when "2026-10-16 18:00" --radius 3.5 --n 5

## Pakai dari bahasa lain
Model ONNX menerima `input` float32 `[N, 28]` dan mengeluarkan `output` `[N, 1]`.
Input = fitur sesuai urutan `features` di metadata, lalu diskalakan `(x - min) / (max - min + 1e-9)` (min/max dari `scaler`).
Jika `target` = `count`, peluang = `1 - exp(-output)`. Logika fitur ada di `feature_engine.extract()` — porting ke bahasa lain harus identik.

## Catatan penting
- Model hanya merekomendasikan area yang ada di data latihnya.
- Nilai `probability` adalah SKOR POTENSI untuk membandingkan area (bukan peluang absolut) dan bukan jaminan order.
- Kalender libur di `feature_engine.py` adalah perkiraan; perbarui bila perlu.
'''


def build_package(onnx_path, out_dir):
    p, out = Path(onnx_path), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = p.with_name(p.stem + "_metadata.json")
    shutil.copy(p, out / p.name); shutil.copy(meta, out / meta.name); shutil.copy(Path(fe.__file__), out / "feature_engine.py")
    (out / "predict_demand.py").write_text(PREDICT_PY.replace("__MODEL__", p.name), "utf-8")
    (out / "README.md").write_text(README.replace("__MODEL__", p.name).replace("__STEM__", p.stem), "utf-8")
    return out


def export_zip(onnx_path, zip_path):
    tmp = Path(tempfile.mkdtemp()); pkg = build_package(onnx_path, tmp / "driver_tracker_model")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in pkg.iterdir(): z.write(f, f"driver_tracker_model/{f.name}")
    shutil.rmtree(tmp, ignore_errors=True)


def test_package(onnx_path):
    """Simulasi uji: paket dijalankan MANDIRI di proses terpisah, hasilnya dibandingkan dengan model di aplikasi."""
    tmp = Path(tempfile.mkdtemp()); pkg = build_package(onnx_path, tmp / "pkg"); rt = tr.Runtime(onnx_path); m = rt.meta; checks = []
    cells = [tuple(map(int, k.split(","))) for k in list(m["cell_rate"])[:30]]; ref = rt.predict(cells, SCEN)
    try:
        r = subprocess.run([sys.executable, str(pkg / "predict_demand.py"), "--selftest"], capture_output=True, text=True, cwd=pkg, timeout=180)
        got = np.array(json.loads(r.stdout)["pred"]); diff = float(np.abs(got - ref).max())
        checks.append(("Paket mandiri = hasil di aplikasi", "PASS" if diff < 1e-4 else "FAIL", f"selisih maks {diff:.1e} ({got.size} prediksi)"))
        r2 = subprocess.run([sys.executable, str(pkg / "predict_demand.py"), "--lat", "-7.2625", "--lon", "112.7381", "--radius", "5"], capture_output=True, text=True, cwd=pkg, timeout=180)
        n = len(json.loads(r2.stdout)); checks.append(("CLI memberi rekomendasi", "PASS" if n else "WARN", f"{n} spot untuk titik uji Tunjungan Plaza"))
    except Exception as e:
        checks.append(("Paket mandiri bisa dijalankan", "FAIL", str((getattr(e, 'stderr', '') or e))[-250:]))
    checks.append(("Skor dalam rentang 0–1", "PASS" if ((ref >= 0) & (ref <= 1)).all() else "FAIL", f"min {ref.min():.3f}, maks {ref.max():.3f}"))
    top = sorted(m["cell_rate"], key=m["cell_rate"].get, reverse=True)[:5]; tc = [tuple(map(int, k.split(","))) for k in top]; pm = rt.predict(tc, SCEN)
    spread = float((pm.max(axis=1) - pm.min(axis=1)).mean())
    checks.append(("Prediksi peka terhadap hari/jam", "PASS" if spread > .02 else "WARN", f"rata-rata selisih antar skenario {spread:.3f}"))
    q, qt = tr.quality_label(m["metrics"]); checks.append(("Kualitas model", "PASS" if q != "Kurang" else "WARN", f"{q} (AUC {m['metrics']['roc_auc']})"))
    mt = m["metrics"]
    bb = mt.get("top5_baseline_best", mt.get("top5_baseline_popular"))
    if bb is not None: lift = mt["top5_hit_rate"] - bb; checks.append(("Dibanding tebakan sederhana terbaik", "PASS" if lift >= .02 else "WARN", f"Top-5 model {mt['top5_hit_rate']:.0%} vs tebakan {bb:.0%}"))
    lines = [f"{'✔' if s == 'PASS' else '⚠' if s == 'WARN' else '✖'} {s}  {n} — {d}" for n, s, d in checks]
    lines += ["", "Simulasi 3 area teramai:", "%-22s" % "" + "".join(f"{n:>11}" for n in SCEN_NAMES)]
    for i in range(min(3, len(tc))):
        cl, co = fe.cell_center(*tc[i], m["grid_km"]); lines.append("%-22s" % fe.nearest_name(cl, co, *tc[i]) + "".join(f"{p * 100:>9.0f}/100" for p in pm[i]))
    shutil.rmtree(tmp, ignore_errors=True)
    return dict(ok=all(s != "FAIL" for _, s, _ in checks), checks=checks, report="\n".join(lines))
