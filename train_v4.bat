@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Training Model v4 Champion (ShuffleNetV2 192 + PosEmb)
echo ========================================================
.venv\Scripts\python.exe baseline\train_v4_champion.py %*
pause
