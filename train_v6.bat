@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Training Model v6 Compact-64 (FT + Sigmoid Distill)
echo ========================================================
.venv\Scripts\python.exe baseline\train_v6_compact64.py %*
pause
