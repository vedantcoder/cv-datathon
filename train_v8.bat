@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Training Model v8 Champion (40px Stage4 + Top-550)
echo ========================================================
.venv\Scripts\python.exe baseline\train_v8_champion.py %*
pause
