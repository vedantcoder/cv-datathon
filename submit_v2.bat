@echo off
echo Packaging and verifying V2 Gated Model submission on CPU (Codabench environment)...
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v2_gated --ckpt data\checkpoints\v2_gated.pt --out submission_v2_gated.zip --n-val 2000 --threads 2
pause
