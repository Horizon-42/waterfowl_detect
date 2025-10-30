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
                device=device, batch=4)
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


def predict_with_tiles(model, image_path, tile_size=640, overlap=0.2, imgsz=640, conf=0.25):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    # convert grayscale or BGRA tiles to 3-channel BGR as expected by YOLO
    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width = image.shape[:2]

    stride = int(tile_size * (1 - overlap))
    boxes = []
    scores = []
    class_ids = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = image[y:y_end, x:x_end]

            results = model.predict(source=tile, imgsz=imgsz, conf=conf)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    conf_score = float(box.conf[0].cpu().item())
                    class_id = int(box.cls[0].cpu().item())

                    # Adjust box coordinates to original image
                    boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    scores.append(conf_score)
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    if boxes:
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)

        indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold=0.5)

        final_boxes = boxes[indices].numpy()
        final_scores = scores[indices].numpy()
        final_class_ids = class_ids[indices].numpy()
    else:
        final_boxes = []
        final_scores = []
        final_class_ids = []

    # draw boxes on image
    result_image = image.copy()
    for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(result_image, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    result_dir = os.path.join("runs", "detect", "predict_tiled")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir, 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, result_image)
    print(f"Result saved to {result_path}")
    return final_boxes, final_scores, final_class_ids


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
                        default='yolo11x.pt')
    parser.add_argument('--test_images', type=str, required=False,
                        help='Path to test images (required if mode is test)',
                        default="datasets/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/03_Negative Images")
    args = parser.parse_args()
    mode = args.mode

    latest_run = get_last_run_directory()
    model_path = os.path.join(latest_run, 'weights', 'best.pt')

    if mode == 'train':
        train_yolo(data_yaml, model_path=args.pretrained_model,
                   epochs=350, imgsz=640)
    elif mode == 'val':
        validate_yolo(model_path=model_path, imgsz=1024)
    else:
        test_yolo(model_path=model_path,
                  test_images=args.test_images)
