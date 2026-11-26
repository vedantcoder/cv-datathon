@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Verifying and Building submission_v6_compact64.zip
echo ========================================================
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v6_compact64 --ckpt data\checkpoints\v6_compact64.pt --out submission_v6_compact64.zip --n-val 2000 --threads 2 %*
echo.
echo ========================================================
echo Done! Upload submission_v6_compact64.zip to Codabench.
echo ========================================================
pause
