@echo off
echo Starting Model v3 (ShuffleNetV2 x0.5) training on NVIDIA RTX 3050 GPU...
.venv\Scripts\python.exe baseline\train_v3_shufflenet.py --epochs 45 --batch-size 128 --lr 2e-3
pause
