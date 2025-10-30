from ultralytics import YOLO
import cv2
import torch
import os
import shutil
import argparse

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


def test_yolo(model_path, test_images, imgsz=640):
    model = YOLO(model_path)
    results = model.predict(
        source=test_images, conf=0.25, save=True)
    print("Test results saved in 'runs/detect/predict' directory.")

if __name__ == "__main__":
    data_yaml = 'data.yaml'  # path to your data.yaml file

    # read argument from command line
    # mode = ['train', 'val', 'test']
    # --test_images path/to/images
    parser = argparse.ArgumentParser(
        description='YOLO Training and Validation')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], required=False,
                        help='Mode to run: train, val, or test')
    parser.add_argument('--pretrained_model', type=str, required=False,
                        help='Path to pretrained model (default: yolov11x.pt)',
                        default='yolov11x.pt')
    parser.add_argument('--test_images', type=str, required=False,
                        help='Path to test images (required if mode is test)',
                        default="datasets/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/03_Negative Images")
    args = parser.parse_args()
    mode = args.mode

    latest_run = get_last_run_directory()
    model_path = os.path.join(latest_run, 'weights', 'best.pt')

    if mode == 'train':
        train_yolo(data_yaml, model_path=args.pretrained_model,
                   epochs=350, imgsz=1024)
    elif mode == 'val':
        validate_yolo(model_path=model_path, imgsz=1024)
    else:
        test_yolo(model_path=model_path,
                  test_images=args.test_images)
