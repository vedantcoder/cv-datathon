@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo   Verifying and Building submission_v5_tiny_ft.zip
echo ========================================================
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v5_tiny_ft --ckpt data\checkpoints\v5_tiny_ft_80.pt --out submission_v5_80.zip --n-val 2000 --threads 2 %*
echo.
echo ========================================================
echo Done! Upload submission_v5_80.zip to Codabench.
echo ========================================================
pause
