@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Verifying and Building submission_v4_champion.zip
echo ========================================================
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v4_champion --ckpt data\checkpoints\v4_champion.pt --out submission_v4_champion.zip --n-val 2000 --threads 2 %*
echo.
echo ========================================================
echo Done! Upload submission_v4_champion.zip to Codabench.
echo ========================================================
pause
