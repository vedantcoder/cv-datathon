@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Verifying and Building submission_v9_mbv3_ft.zip
echo ========================================================
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v9_mbv3_ft --ckpt data\checkpoints\v9_mbv3_ft.pt --out submission_v9_mbv3_ft.zip --n-val 2000 --threads 2 %*
echo.
echo ========================================================
echo Done! Verify Gap <= 20 before uploading to Codabench.
echo ========================================================
pause
