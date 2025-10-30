from ultralytics import YOLO
import cv2
import torch
import os
import shutil

def get_last_run_directory(base_dir='runs/detect'):
    import re
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    # only select train dirs
    runs = [d for d in runs if re.match(r'train\d+', os.path.basename(d))]

    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def train_yolo(data_yaml, model_path='yolov8n.pt', epochs=50, imgsz=640):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                device=device, batch=4, resume=True)
    # move the best model to 'trained.pt'

    latest_run = get_last_run_directory()
    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
    print(f"Latest run directory: {latest_run}")
    shutil.copy(best_model_path, 'trained.pt')


def draw_loss_png(loss_history_path, output_path='loss_curve.png'):
    import matplotlib.pyplot as plt
    import pandas as pd

    loss_history = pd.read_csv(loss_history_path)

    # remove time column if exists
    if 'time' in loss_history.columns:
        loss_history = loss_history.drop(columns=['time'])

    plt.figure(figsize=(10, 6))
    # draw every loss curve
    i = 1
    for column in ["train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if column != 'epoch':
            # plot on subplot
            plt.subplot(2, 5, i)
            i += 1
            plt.plot(loss_history['epoch'], loss_history[column], label=column)
            plt.title(column)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def validate_yolo(model_path, imgsz=640):
    model = YOLO(model_path)
    results = model.val(imgsz=imgsz, batch=4)
    # plot confusion matrix
    # results.plot()

    train_dir = os.path.dirname(os.path.dirname(model_path))
    draw_loss_png(os.path.join(train_dir, 'results.csv'),
                  output_path=os.path.join(train_dir, 'results.png'))

    # move model
    best_model_path = os.path.join(train_dir, 'weights', 'best.pt')
    print(f"Latest run directory: {train_dir}")
    shutil.copy(best_model_path, 'trained.pt')

if __name__ == "__main__":
    data_yaml = 'data.yaml'  # path to your data.yaml file

    # select trian and val mode
    # read parameter from command line or set manually
    mode = input("Enter mode (train/val/test): ").strip().lower()
    if mode == 'val':
        latest_run = get_last_run_directory()
        model_path = os.path.join(latest_run, 'weights', 'best.pt')
        validate_yolo(model_path=model_path, imgsz=1024)
    elif mode == 'train':
        train_yolo(data_yaml, model_path='runs/detect/train12/weights/best.pt',
                   epochs=350, imgsz=1024)
    else:
        print("Invalid mode. Please enter 'train' or 'val'.")
