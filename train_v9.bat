@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Training Model v9 (End-to-End MobileNetV3-Small + KD)
echo ========================================================
.venv\Scripts\python.exe baseline\train_v9_mbv3_ft.py %*
pause
