@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Training Model v5 (End-to-End Fine-Tuned + Distilled)
echo ========================================================
.venv\Scripts\python.exe baseline\train_v5_tiny_ft.py %*
pause
