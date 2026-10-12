"""28 fitur spasiotemporal + kalender Indonesia/Jawa Timur + generator data demo."""
import math, random, json, datetime as dt
from functools import lru_cache

FEATURES = ["grid_x", "grid_y", "order_hour_sin", "order_hour_cos",
            "slot_pagi", "slot_siang", "slot_sore", "slot_malam", "slot_larut_malam",
            "is_senin", "is_selasa", "is_rabu", "is_kamis", "is_jumat", "is_sabtu", "is_minggu",
            "is_monday_rush", "is_friday_early_rush", "is_friday_evening_hangout",
            "is_holiday", "is_school_holiday", "is_college_holiday", "is_payday",
            "cell_historical_order_rate", "cell_dropoff_inflow_rate", "cell_slot_rate", "cell_dow_rate", "neighbor_rate"]
assert len(FEATURES) == 28

HOTSPOTS = {  # nama: (lat, lon, kategori)
    "Tunjungan Plaza": (-7.2625, 112.7381, "mall"), "Stasiun Gubeng": (-7.2655, 112.7523, "transport"),
    "Pakuwon Mall": (-7.2893, 112.6726, "mall"), "Kampus ITS": (-7.2819, 112.7948, "campus"),
    "SIER Rungkut": (-7.3404, 112.7623, "office"), "Stasiun Pasar Turi": (-7.2455, 112.7288, "transport"),
    "Bandara Juanda": (-7.3798, 112.7869, "transport"), "Galaxy Mall": (-7.2756, 112.7803, "mall"),
    "Kenjeran Park": (-7.2494, 112.7954, "kuliner"), "Kampus Unair C": (-7.2679, 112.7861, "campus")}

# Hari libur nasional dan cuti bersama resmi Indonesia (SKB 3 Menteri).
# Cuti bersama ikut ditandai agar fitur kalender/payday tidak memperlakukannya
# sebagai hari kerja biasa. Perbarui set ini saat SKB tahun berikutnya terbit.
HOLIDAYS = {
    # 2025: SKB Perubahan No. 933 Tahun 2025 / No. 1 Tahun 2025 / No. 3 Tahun 2025
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-03-28", "2025-03-29",
    "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07",
    "2025-04-18", "2025-04-20", "2025-05-01", "2025-05-12", "2025-05-13", "2025-05-29",
    "2025-05-30", "2025-06-01", "2025-06-06", "2025-06-09", "2025-08-17", "2025-08-18",
    "2025-09-05", "2025-12-25", "2025-12-26",
    # 2026: SKB resmi 17 hari libur nasional dan 8 cuti bersama
    "2026-01-01", "2026-01-16", "2026-02-16", "2026-02-17", "2026-03-18", "2026-03-19",
    "2026-03-20", "2026-03-21", "2026-03-22", "2026-03-23", "2026-03-24", "2026-04-03",
    "2026-04-05", "2026-05-01", "2026-05-14", "2026-05-15", "2026-05-27", "2026-05-28",
    "2026-05-31", "2026-06-01", "2026-06-16", "2026-08-17", "2026-08-25", "2026-12-24",
    "2026-12-25",
    # 2027: SKB resmi 18 hari libur nasional dan 8 cuti bersama
    "2027-01-01", "2027-01-05", "2027-02-05", "2027-02-06", "2027-03-08", "2027-03-09",
    "2027-03-10", "2027-03-11", "2027-03-12", "2027-03-15", "2027-03-25", "2027-03-26",
    "2027-03-28", "2027-05-01", "2027-05-06", "2027-05-17", "2027-05-18", "2027-05-19",
    "2027-05-20", "2027-06-01", "2027-06-06", "2027-08-15", "2027-08-17", "2027-12-24",
    "2027-12-25", "2027-12-26",
}
LAT0, LON0 = -7.55, 112.55
DAYS = ["senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]


def _deg(km):
    return km / 111.0, km / (111.0 * math.cos(math.radians(7.3)))


def cell_of(lat, lon, km=1.0):
    a, b = _deg(km)
    return int((lat - LAT0) // a), int((lon - LON0) // b)


def cell_center(gx, gy, km=1.0):
    a, b = _deg(km)
    return LAT0 + (gx + .5) * a, LON0 + (gy + .5) * b


def hav(la1, lo1, la2, lo2):
    p = math.pi / 180
    h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def nearest_name(lat, lon, gx, gy):
    n, d = min(((n, hav(lat, lon, v[0], v[1])) for n, v in HOTSPOTS.items()), key=lambda x: x[1])
    return n if d < 1.5 else f"Grid {gx},{gy}"


def is_holiday(d): return d.isoformat() in HOLIDAYS


@lru_cache(None)
def paydays(y, m):
    last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(1)).day
    out = set()
    for day in (1, 25, last):
        d = dt.date(y, m, day)
        while d.weekday() >= 5 or is_holiday(d):  # geser mundur bila libur/akhir pekan
            d -= dt.timedelta(1)
        out.add(d)
    return out


def is_payday(d):
    ny, nm = (d.year + (d.month == 12), d.month % 12 + 1)
    return d in paydays(d.year, d.month) or d in paydays(ny, nm)


def is_school_holiday(d):  # perkiraan libur sekolah Jawa Timur
    md = (d.month, d.day)
    return (6, 27) <= md <= (7, 12) or md >= (12, 22) or md <= (1, 3)


def is_college_holiday(d):  # perkiraan libur semester kampus
    md = (d.month, d.day)
    return (7, 15) <= md <= (8, 31) or (1, 10) <= md <= (2, 20)


def slot_of(h): return 0 if 5 <= h < 11 else 1 if 11 <= h < 15 else 2 if 15 <= h < 18 else 3 if 18 <= h < 22 else 4


def get_priors(meta, gx, gy, when):
    """Prior sel untuk prediksi: (rate/hari, inflow/hari, rate slot-jam, rate hari-minggu, rate tetangga)."""
    k = f"{gx},{gy}"; s = slot_of(when.hour + when.minute / 60)
    return (meta["cell_rate"].get(k, 0), meta["cell_inflow"].get(k, 0), meta["cell_slot"].get(k, [0] * 5)[s],
            meta["cell_dow"].get(k, [0] * 7)[when.weekday()], meta["cell_nb"].get(k, 0))


def extract(gx, gy, when, pri=(0, 0, 0, 0, 0), opts=None):
    o = opts or {}; d, wd = when.date(), when.weekday(); h = when.hour + when.minute / 60; cal = o.get("calendar", True)
    f = [gx / 100, gy / 100, math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]
    f += [1.0 if slot_of(h) == i else 0.0 for i in range(5)]
    f += [1.0 if wd == i else 0.0 for i in range(7)]
    f += [float(wd == 0 and (6.5 <= h <= 9.5 or 16.5 <= h <= 19.5)), float(wd == 4 and 10.5 <= h <= 12.75), float(wd == 4 and 16.5 <= h <= 22)]
    f += [float(cal and is_holiday(d)), float(cal and is_school_holiday(d)), float(cal and is_college_holiday(d))]
    f += [float(o.get("payday", True) and is_payday(d))] + [float(x) for x in pri]
    return f


def flags(when, opts=None):
    """Badge indikator aktif untuk simulasi."""
    f = dict(zip(FEATURES, extract(0, 0, when, opts=opts)))
    out = [f"is_{DAYS[when.weekday()]}=1"]
    for k, t in (("is_monday_rush", "Senin Rush Hour"), ("is_friday_early_rush", "Jumat <12:00"),
                 ("is_friday_evening_hangout", "Jumat Malam Hangout"), ("is_holiday", "Libur Nasional"),
                 ("is_school_holiday", "Libur Sekolah"), ("is_payday", "Gajian")):
        if f[k]: out.append(t)
    return out


BOX = (-7.65, -7.05, 112.45, 113.0)  # Surabaya raya (+Sidoarjo/Gresik): lat_min, lat_max, lon_min, lon_max. Ubah bila kotamu lain.


def _f(x): return float(str(x).strip().replace(",", "."))


def norm_jenis(s):
    s = str(s).strip().lower()
    if any(k in s for k in ("food", "makan", "resto", "kuliner", "mart")): return "Food"
    if any(k in s for k in ("send", "kirim", "kurir", "paket", "express", "instant")): return "Send"
    if any(k in s for k in ("ride", "bike", "car", "motor", "mobil", "ojek", "antar", "taxi")): return "Ride"
    return "Lainnya"


def parse_order(o, km=1.0):
    """Validasi + normalisasi 1 record. Return (record, None) atau (None, alasan_ditolak)."""
    try:
        d = dt.date.fromisoformat(str(o["tanggal"]).strip()[:10]); p = str(o["jamMulai"]).strip().split(":"); h, m = int(p[0]), int(p[1])
        if not (0 <= h < 24 and 0 <= m < 60): return None, "jam tidak valid"
        la, lo = _f(o["latitudePickup"]), _f(o["longitudePickup"])
    except KeyError as e: return None, f"kolom hilang: {e.args[0]}"
    except Exception: return None, "format tanggal/jam/koordinat tidak valid"
    if d > dt.date.today() + dt.timedelta(1): return None, "tanggal di masa depan"
    if not (BOX[0] <= la <= BOX[1] and BOX[2] <= lo <= BOX[3]): return None, "titik jemput di luar wilayah Surabaya & sekitar"
    try:
        ea, eo = _f(o["latitudeAkhir"]), _f(o["longitudeAkhir"])
        if not (BOX[0] <= ea <= BOX[1] and BOX[2] <= eo <= BOX[3]) or hav(la, lo, ea, eo) > 80: raise ValueError
    except Exception: ea, eo = la, lo  # tujuan kosong/aneh -> diabaikan (bukan membuang order)
    return dict(when=dt.datetime(d.year, d.month, d.day, h, m), cell=cell_of(la, lo, km), drop=cell_of(ea, eo, km),
                jenis=norm_jenis(o.get("jenisOrder", "")), lat=la, lon=lo, end_lat=ea, end_lon=eo), None


def _k(r):
    # Include destination so separate same-minute trips from one pickup aren't collapsed.
    return (r["when"], round(r["lat"], 5), round(r["lon"], 5), round(r.get("end_lat", r["lat"]), 5),
            round(r.get("end_lon", r["lon"]), 5), r["jenis"])


def clean_orders(records, km=1.0):
    """Pembersihan sebelum pelatihan: validasi, wilayah, tanggal, normalisasi jenis, buang duplikat. Return (list rec terurut, {alasan: jumlah})."""
    stats, seen, out = {}, set(), []
    for o in records:
        r, e = parse_order(o, km)
        if not r: stats[e] = stats.get(e, 0) + 1; continue
        if _k(r) in seen: stats["duplikat"] = stats.get("duplikat", 0) + 1; continue
        seen.add(_k(r)); out.append(r)
    return sorted(out, key=lambda r: r["when"]), stats


def demo_orders(days=70, seed=7):
    rnd = random.Random(seed)
    names = list(HOTSPOTS)
    today, out = dt.date.today(), []
    for k in range(days, 0, -1):
        d = today - dt.timedelta(k)
        wd = d.weekday()
        for _ in range(rnd.randint(22, 34) + (8 if wd in (0, 4) else 0)):
            hr = rnd.choices(range(24), [1, 1, 0.5, 0.3, 0.3, 1, 3, 6, 6, 4, 3, 4, 5, 4, 3, 3, 4, 6, 6, 5, 4, 3, 2, 1.5])[0]
            ws = []
            for n in names:
                c, w = HOTSPOTS[n][2], 1.0
                if c in ("transport", "office") and (6 <= hr <= 9 or 16 <= hr <= 19) and wd < 5: w = 3
                if c == "mall" and (17 <= hr <= 21 or wd >= 5): w = 3 + (2 if wd == 4 and hr >= 16 else 0)
                if c == "campus" and 7 <= hr <= 15 and wd < 5: w = 3
                if c == "kuliner" and hr >= 17: w = 2.5 + (2 if wd == 4 else 0)
                if n == "Bandara Juanda": w *= .6
                ws.append(w)
            n = rnd.choices(names, ws)[0]
            la, lo, _ = HOTSPOTS[n]
            dn = rnd.choice(names)
            out.append({"latitudePickup": round(la + rnd.gauss(0, .0035), 6), "longitudePickup": round(lo + rnd.gauss(0, .0035), 6),
                        "latitudeAkhir": round(HOTSPOTS[dn][0] + rnd.gauss(0, .004), 6),
                        "longitudeAkhir": round(HOTSPOTS[dn][1] + rnd.gauss(0, .004), 6),
                        "jamMulai": f"{hr:02d}:{rnd.randint(0, 59):02d}", "tanggal": d.isoformat(),
                        "jenisOrder": rnd.choices(["Ride", "Food", "Send"], [5, 4, 1])[0]})
    return out


JENIS_COLOR = {"Ride": "#16A34A", "Food": "#EF4444", "Send": "#2563EB"}
JENIS_LABEL = {"Ride": "Ride", "Food": "Food", "Send": "Paket"}


def load_json_records(path):
    """Baca file .json ATAU .csv (kolom sama dengan JSON)."""
    if str(path).lower().endswith(".csv"):
        import csv
        return [dict(r) for r in csv.DictReader(open(path, encoding="utf-8-sig", newline=""))]
    d = json.load(open(path, encoding="utf-8-sig"))
    if isinstance(d, dict):
        lists = [v for v in d.values() if isinstance(v, list)]
        d = lists[0] if lists else [d]
    return [x for x in d if isinstance(x, dict)]


def merge_orders(store, records):
    """Gabungkan ke store (in-place): record dinormalisasi, duplikat dibuang. Return (baru, duplikat, daftar_alasan_ditolak)."""
    seen = {_k(r) for r in (parse_order(o)[0] for o in store) if r}
    added = dup = 0; bad = []
    for o in records:
        r, e = parse_order(o)
        if not r: bad.append(e); continue
        if _k(r) in seen: dup += 1; continue
        seen.add(_k(r)); n = dict(o)
        n.update(tanggal=r["when"].date().isoformat(), jamMulai=r["when"].strftime("%H:%M"), latitudePickup=r["lat"], longitudePickup=r["lon"],
                 latitudeAkhir=r["end_lat"], longitudeAkhir=r["end_lon"], jenisOrder=r["jenis"])
        store.append(n); added += 1
    return added, dup, bad


def template_orders():
    return [{"latitudePickup": -7.2625, "longitudePickup": 112.7381, "latitudeAkhir": -7.2655, "longitudeAkhir": 112.7523,
             "jamMulai": "08:15", "tanggal": "2026-09-01", "jenisOrder": "Ride"},
            {"latitudePickup": -7.2893, "longitudePickup": 112.6726, "latitudeAkhir": -7.2756, "longitudeAkhir": 112.7803,
             "jamMulai": "18:40", "tanggal": "2026-09-01", "jenisOrder": "Food"},
            {"latitudePickup": -7.2819, "longitudePickup": 112.7948, "latitudeAkhir": -7.2679, "longitudeAkhir": 112.7861,
             "jamMulai": "12:05", "tanggal": "2026-09-02", "jenisOrder": "Send"}]


def data_report(store):
    ds, cells, n = set(), set(), 0
    for o in store:
        r, _ = parse_order(o)
        if r: n += 1; ds.add(r["when"].date()); cells.add(r["cell"])
    ready, tips = n >= 100 and len(ds) >= 7, []
    if n == 0: tips.append("• Mulai dengan impor file JSON/CSV order Anda (tombol Unduh Template menunjukkan formatnya).")
    else:
        tips.append(f"• Data mencakup {len(ds)} hari berbeda di {len(cells)} area.")
        if not ready: tips.append("• Syarat minimum pelatihan: 100 order dan 7 hari berbeda.")
        if n < 300: tips.append("• Idealnya ≥ 300 order agar prediksi stabil.")
        if len(ds) < 14: tips.append("• Kumpulkan ≥ 14 hari agar pola hari (Senin/Jumat) terbaca.")
        if len(cells) < 5: tips.append("• Area masih sedikit; model hanya bisa merekomendasikan area yang ada di data.")
        if n >= 300 and len(ds) >= 14 and len(cells) >= 5: tips.append("✔ Data sudah cukup baik untuk dilatih.")
    return dict(n=n, days=len(ds), areas=len(cells), ready=ready, tips=tips)
