import os, sys, ctypes
from ultralytics import YOLO

os.chdir(r"C:\Users\User\Desktop\ESP")

# Redirect console output to a file so IDLE doesn't freeze on tqdm progress bars
log = open("train_log.txt", "w", encoding="utf-8", buffering=1)
sys.stdout = log
sys.stderr = log

YOLO("yolov8n.pt").train(
    data="dataset.yaml",
    imgsz=416,
    epochs=25,
    batch=16,
    device=0,
    workers=0,
    cache=True,
    plots=False,
)

log.close()
ctypes.windll.user32.MessageBoxW(0, "Training complete! Copy best.pt from runs\\detect\\train-7\\weights\\", "YOLO Train", 0)
