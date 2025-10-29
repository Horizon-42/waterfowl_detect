from ultralytics import YOLO
import cv2
import torch

def train_yolo(data_yaml, model_path='yolov8n.pt', epochs=50, imgsz=640):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device, batch=4)
    # move the best model to 'trained.pt'

    import shutil
    # first get the newest run directory
    import os
    runs_dir = 'runs/detect'
    latest_run = max([os.path.join(runs_dir, d)
                     for d in os.listdir(runs_dir)], key=os.path.getmtime)
    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
    print(f"Latest run directory: {latest_run}")
    shutil.copy(best_model_path, 'trained.pt')

if __name__ == "__main__":
    data_yaml = 'data.yaml'  # path to your data.yaml file
    train_yolo(data_yaml, model_path='yolo11m.pt', epochs=100, imgsz=1024)