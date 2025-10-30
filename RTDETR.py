from ultralytics import RTDETR
import cv2
import torch
import os
import shutil
import argparse

from utils import get_last_run_directory, predict_with_tiles, draw_loss_png



def train_rtdetr(data_yaml, model_path='rtdetr-l.pt', epochs=50, imgsz=640, batch=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = RTDETR(model_path)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                device=device, batch=batch)

    latest_run = get_last_run_directory()
    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
    print(f"Latest run directory: {latest_run}")
    shutil.copy(best_model_path, 'trained_rtdetr.pt')



def validate_rtdetr(model_path, imgsz=640, batch=4):
    model = RTDETR(model_path)
    model.val(imgsz=imgsz, batch=batch)

    train_dir = os.path.dirname(os.path.dirname(model_path))
    results_csv = os.path.join(train_dir, 'results.csv')
    if os.path.exists(results_csv):
        draw_loss_png(results_csv, output_path=os.path.join(train_dir, 'results.png'))

    best_model_path = os.path.join(train_dir, 'weights', 'best.pt')
    print(f"Latest run directory: {train_dir}")
    shutil.copy(best_model_path, 'trained_rtdetr.pt')


def predict_with_tiles(model, image_path, tile_size=640, overlap=0.2, imgsz=640, conf=0.25):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    if image.ndim == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
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

                    boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    scores.append(conf_score)
                    class_ids.append(class_id)

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

    result_image = image.copy()
    for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(result_image, f"{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    result_dir = os.path.join("runs", "detect", "predict_tiled_rtdetr")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, result_image)
    print(f"Result saved to {result_path}")
    return final_boxes, final_scores, final_class_ids


def test_rtdetr(model_path, test_images, imgsz=640, tile_size=640):
    model = RTDETR(model_path)
    if tile_size == 0:
        model.predict(source=test_images, conf=0.25, imgsz=imgsz)
    else:
        if os.path.isdir(test_images):
            image_files = [os.path.join(test_images, f) for f in sorted(os.listdir(test_images))
                           if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
            for image_file in image_files:
                predict_with_tiles(model, image_file,
                                   tile_size=tile_size, overlap=0.2, imgsz=imgsz, conf=0.25)
    print("Test results saved for RTDETR predictions.")


if __name__ == "__main__":
    data_yaml = 'data.yaml'

    parser = argparse.ArgumentParser(description='RT-DETR Training and Validation')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], required=False,
                        help='Mode to run: train, val, or test')
    parser.add_argument('--pretrained_model', type=str, required=False,
                        help='Path to pretrained RT-DETR model (default: rtdetr-l.pt)',
                        default='rtdetr-l.pt')
    parser.add_argument('--test_images', type=str, required=False,
                        help='Path to test images (required if mode is test)',
                        default="datasets/46k66mz9sz-2/00_UAV-derived Thermal Waterfowl Dataset/00_UAV-derived Waterfowl Thermal Imagery Dataset/01_Thermal Images and Ground Truth (used for detector training and testing)/03_Negative Images")
    parser.add_argument('--epochs', type=int, default=350,
                        help='Number of epochs for training (default: 350)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training/validation (default: 640)')
    parser.add_argument('--batch', type=int, default=4,
                        help='Batch size (default: 4)')
    args = parser.parse_args()
    mode = args.mode

    try:
        latest_run = get_last_run_directory()
        model_path = os.path.join(latest_run, 'weights', 'best.pt')
    except FileNotFoundError:
        latest_run = None
        model_path = None

    if mode == 'train':
        train_rtdetr(data_yaml, model_path=args.pretrained_model,
                     epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    elif mode == 'val':
        if model_path is None:
            raise FileNotFoundError('No trained RT-DETR model found for validation.')
        validate_rtdetr(model_path=model_path, imgsz=args.imgsz, batch=args.batch)
    else:
        if model_path is None:
            raise FileNotFoundError('No trained RT-DETR model found for testing.')
        test_rtdetr(model_path=model_path,
                    test_images=args.test_images, imgsz=args.imgsz, tile_size=640)
