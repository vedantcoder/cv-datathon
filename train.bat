@echo off
echo Starting MobileNetV3-Small training on NVIDIA RTX 3050 GPU...
.venv\Scripts\python.exe baseline\train_mbv3_small.py --epochs 45 --batch-size 128 --lr 2e-3
pause
