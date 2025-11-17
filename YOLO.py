from ultralytics import YOLO
import cv2
import torch
import os
import shutil
import argparse

from utils import get_last_run_directory, predict_with_tiles, draw_loss_png


def train_yolo(data_yaml, model_path='yolov8n.pt', epochs=50, imgsz=640):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLO(model_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                device=device, batch=8, dropout = 0.2, plots = False,
                degrees = 10.0,
                translate = 0.1,
                flipud = 0.5,
                fliplr = 0.5,
                mosaic= 0.5,
                copy_paste = 0.5,
                resume = True,
                patience = 20
                )
    # move the best model to 'trained.pt'

    latest_run = get_last_run_directory()
    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
    print(f"Latest run directory: {latest_run}")
    shutil.copy(best_model_path, 'trained.pt')


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



def test_yolo(model_path, test_images, imgsz=640, tile_size=640):
    model = YOLO(model_path)
    if tile_size == 0:
        results = model.predict(
            source=test_images, conf=0.25, imgsz=imgsz)
    else:
        # get all the images in the test_images directory
        if os.path.isdir(test_images):
            image_files = [os.path.join(test_images, f) for f in os.listdir(
                test_images) if f.lower().endswith(('.tif', '.tiff'))]
            for image_file in image_files:
                predict_with_tiles(model, image_file,
                                   tile_size=tile_size, overlap=0.2, imgsz=imgsz, conf=0.25)
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
                        help='Path to pretrained model (default: yolo11x.pt)',
                        default='yolo11l.pt')
    parser.add_argument('--test_images', type=str, required=False,
                        help='Path to test images (required if mode is test)',
                        default="datasets/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/03_Negative Images")
    args = parser.parse_args()
    mode = args.mode

    latest_run = "runs/detect/train10"
    model_path = os.path.join(latest_run, 'weights', 'best.pt')

    if mode == 'train':
        train_yolo(data_yaml, model_path=args.pretrained_model,
                   epochs=100, imgsz=512)
    elif mode == 'val':
        validate_yolo(model_path=model_path, imgsz=512)
    else:
        test_yolo(model_path=model_path,
                  test_images=args.test_images)
