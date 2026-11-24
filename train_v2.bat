@echo off
echo Starting V2 Gated Model training on NVIDIA RTX 3050 GPU...
.venv\Scripts\python.exe baseline\train_v2_gated.py --epochs 45 --batch-size 128 --lr 2e-3
pause
