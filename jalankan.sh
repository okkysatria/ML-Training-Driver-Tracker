#!/usr/bin/env bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3 belum terpasang (https://www.python.org/downloads/)"; exit 1; }
[ -d venv ] || { echo "Membuat lingkungan Python (sekali saja)..."; python3 -m venv venv; }
source venv/bin/activate
python -m pip install -r requirements.txt
python cek_instalasi.py && python main.py
