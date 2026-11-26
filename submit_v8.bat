@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Verifying and Building submission_v8_champion.zip
echo ========================================================
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v8_champion --ckpt data\checkpoints\v8_champion.pt --out submission_v8_champion.zip --n-val 2000 --threads 2 %*
echo.
echo ========================================================
echo Done! Upload submission_v8_champion.zip to Codabench.
echo ========================================================
pause
