@echo off
echo Packaging and verifying submission on CPU (Codabench environment)...
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\mbv3_small --ckpt data\checkpoints\mbv3_small.pt --out submission_mbv3_small.zip --n-val 2000 --threads 2
pause
