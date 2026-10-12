@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "PYTHON="
where python >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYTHON (echo Python belum ditemukan. Pasang Python 3.12 atau tambahkan Python ke PATH. & pause & exit /b 1)
"%PYTHON%" -c "import PySide6, numpy, onnx, onnxruntime" >nul 2>&1
if not errorlevel 1 (
  echo Pustaka aplikasi sudah tersedia. Memakai Python yang terpasang.
  "%PYTHON%" cek_instalasi.py
  if errorlevel 1 (pause & exit /b 1)
  "%PYTHON%" main.py
  pause
  exit /b
)
if not exist venv\Scripts\python.exe (echo Membuat lingkungan Python, pertama kali. & "%PYTHON%" -m venv venv)
call venv\Scripts\activate
echo Memasang library (pertama kali agak lama)...
python -m pip install -r requirements.txt || (echo Gagal memasang library. Periksa koneksi internet dan versi Python. & pause & exit /b 1)
python cek_instalasi.py || (pause & exit /b 1)
python main.py
pause
