@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [INFO] Creez virtual environment...
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install pandas openpyxl numpy requests streamlit rapidfuzz
for /f "tokens=2 delims=:," %%a in ('findstr /i "ui_port" config.json') do set PORT=%%a
if "%PORT%"=="" set PORT=8501
set PORT=%PORT: =%
python -m streamlit run app.py --server.address 0.0.0.0 --server.port %PORT%
endlocal
