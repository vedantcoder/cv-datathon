@echo off
echo Packaging and verifying Model v3 (ShuffleNetV2 x0.5) submission on CPU (Codabench environment)...
.venv\Scripts\python.exe eval\check_submission.py --model-dir baseline\v3_shufflenet --ckpt data\checkpoints\v3_shufflenet.pt --out submission_v3_shufflenet.zip --n-val 2000 --threads 2
pause
