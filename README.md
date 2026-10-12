# ML Training Driver Tracker — cara menjalankan

1. **Ekstrak SEMUA isi zip** ke satu folder (jangan jalankan dari dalam zip, jangan ambil file satu-satu).
2. **Windows:** klik dua kali `jalankan.bat` · **Mac/Linux:** `bash jalankan.sh`
   (pertama kali memasang library, beberapa menit; setelah itu langsung buka).
3. Kalau tidak jalan: `python cek_instalasi.py` → keluarannya menunjukkan persis apa yang kurang. Kirim keluarannya (dan `error_log.txt` bila ada).

Perlu: Python 3.10 – 3.12 (disarankan 3.12) dan internet (peta OpenStreetMap + pemasangan pertama).

| Pesan error | Artinya | Perbaikan |
|---|---|---|
| `No module named 'home_view'` (atau file lain) | file tidak lengkap / campuran versi | ekstrak ulang zip lengkap ke folder baru |
| `No module named 'PySide6'` / `onnxruntime` | library belum terpasang | `pip install -r requirements.txt` |
| gagal memasang `onnx`/`onnxruntime` | Python terlalu baru | pasang Python 3.12 |
| jendela terbuka tapi peta abu-abu | tidak ada internet / diblokir | cek koneksi; fitur lain tetap jalan |
