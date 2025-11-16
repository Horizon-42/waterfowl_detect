from PIL import Image
import numpy as np
import torch
from torchvision.models.detection import FasterRCNN
import cv2
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn
from utils import read_test_annotations, compute_metrics
import tqdm

# load model
def load_faster_rcnn_model(num_classes, model_path=None, device='cpu', model_type='v2'):
    if model_type == 'v2':
        model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def predict_with_tiles(model, image_path:str, tile_size=640, overlap=0.2, imgsz=640, conf=0.25):
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

    for y in tqdm.tqdm(range(0, height, stride)):
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = image[y:y_end, x:x_end]

            # convert tile to CHW tensor
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tile = tile.unsqueeze(0)  # add batch dimension
            results = model(tile)
            for result in results:
                for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
                    x1, y1, x2, y2 = box.cpu().tolist()
                    conf_score = score.cpu().item()
                    class_id = label.cpu().item()

                    if conf_score < conf:
                        continue

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

    result_dir = os.path.dirname(image_path)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir, 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, result_image)
    print(f"Result saved to {result_path}")
    return final_boxes, final_scores, final_class_ids


if __name__ == "__main__":
    model_path = "runs/fasterrcnn_4/best_model.pt"
    test_image_path = "datasets/test/test_image.tif"
    label_path = "datasets/test/test_image_groundtruth.csv"

    model = load_faster_rcnn_model(num_classes=2, model_path=model_path, model_type="v2")
    final_boxes, final_scores, final_class_ids = predict_with_tiles(model, test_image_path, tile_size=640, overlap=0.2, imgsz=640, conf=0.25)
    ground_truth_boxes = read_test_annotations(label_path)

    metrics = compute_metrics(final_boxes, final_scores, ground_truth_boxes, iou_threshold=0.3)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
