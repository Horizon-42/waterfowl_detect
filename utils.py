import os
import cv2
import torch
import numpy as np
import pandas as pd

def get_last_run_directory(base_dir='runs/detect'):
    import re
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    # only select train dirs
    runs = [d for d in runs if re.match(r'train\d+', os.path.basename(d))]

    latest_run = max(runs, key=os.path.getmtime)
    return latest_run


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

def iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes, in continuous XYXY format.
    """
    # intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def compute_metrics(predicted_boxes, confidence_scores, ground_truth_boxes, iou_threshold=0.5):
    """
    Only use for binay detection
    """
    TP = 0
    FP = 0

    # sort predicted boxes with confidence_scores
    sorted_idxes = np.argsort(-np.array(confidence_scores))
    predicted_boxes = [predicted_boxes[i] for i in sorted_idxes]
    confidence_scores = [confidence_scores[i] for i in sorted_idxes]

    matched_gt = set()
    for pred_box in predicted_boxes:
        matched_idx = -1
        max_iou = iou_threshold
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue
            iou_res = iou(pred_box, gt_box)
            if iou_res >= max_iou:
                matched_idx = gt_idx
                max_iou = iou_res
        if matched_idx >= 0:
            TP += 1
            matched_gt.add(matched_idx)
        else:
            FP += 1
    FN = len(ground_truth_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'TP': TP, 'FP': FP, 'FN': FN, 'precision': precision, 'recall': recall, 'f1_score': f1_score}


def read_test_annotations(annotation_path, input_size:tuple, traget_size:tuple):
    """
    Read test annotations from a CSV file.
    The CSV file is expected to have columns: imageFilename,x(column),y(row),width,height
    return boxes list in XYXY format
    """
    import pandas as pd
    annotations = pd.read_csv(annotation_path)
    boxes = []
    src_h, src_w = input_size
    dst_h, dst_w = traget_size
    
    pre_scale_x = src_w / dst_w
    suppose_h = src_h / pre_scale_x
    h_to_add = suppose_h - dst_h
    scale = dst_w/src_w
    
    print("Original Size:", input_size)
    print("Target Size:", traget_size)
    print("Pre Scale X:", pre_scale_x)
    print("Supposed Height:", suppose_h)
    print("Height to add:", h_to_add)
    for _, row in annotations.iterrows():
        x1, y1, w, h = row["x(column)"], row["y(row)"], row["width"], row["height"]
        x2, y2 = x1+w, y1+h
        boxes.append((x1*scale, y1*scale, x2*scale, y2*scale))
    return boxes