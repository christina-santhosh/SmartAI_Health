@echo off
setlocal
cd /d "%~dp0"
title Smart AI Health
if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found. Create it with:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
