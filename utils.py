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
    cv2.imwrite(result_path.replace(".tif",".png"), result_image)
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

import numpy as np
import pandas as pd

def compute_iou_matrix(predicted_boxes, ground_truth_boxes):
    """
    Vectorized IoU for M predicted boxes × N GT boxes.
    Returns an (M, N) IoU matrix.
    """
    if len(predicted_boxes) == 0 or len(ground_truth_boxes) == 0:
        return np.zeros((len(predicted_boxes), len(ground_truth_boxes)))

    pb = predicted_boxes[:, None, :]   # shape M×1×4
    gt = ground_truth_boxes[None, :, :]  # shape 1×N×4

    # Intersection
    xA = np.maximum(pb[..., 0], gt[..., 0])
    yA = np.maximum(pb[..., 1], gt[..., 1])
    xB = np.minimum(pb[..., 2], gt[..., 2])
    yB = np.minimum(pb[..., 3], gt[..., 3])

    inter_w = np.maximum(xB - xA, 0)
    inter_h = np.maximum(yB - yA, 0)
    inter_area = inter_w * inter_h

    # Areas
    pred_area = (pb[..., 2] - pb[..., 0]) * (pb[..., 3] - pb[..., 1])
    gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])

    union = pred_area + gt_area - inter_area
    return inter_area / np.maximum(union, 1e-9)


def compute_metrics_multi_iou_fast(predicted_boxes, confidence_scores, ground_truth_boxes, iou_thresholds=[0.5]):

    predicted_boxes = np.array(predicted_boxes)
    ground_truth_boxes = np.array(ground_truth_boxes)
    confidence_scores = np.array(confidence_scores)

    # Sort by confidence descending
    if len(confidence_scores) > 0:
        sort_idx = np.argsort(-confidence_scores)
        predicted_boxes = predicted_boxes[sort_idx]
        confidence_scores = confidence_scores[sort_idx]

    # Pre-compute IoU matrix: shape = (M_pred, N_gt)
    iou_mat = compute_iou_matrix(predicted_boxes, ground_truth_boxes)

    results = []

    for thr in iou_thresholds:

        TP = 0
        matched_gt = np.zeros(len(ground_truth_boxes), dtype=bool)

        # For each prediction, choose the best GT ≥ threshold
        for i in range(len(predicted_boxes)):
            # IoUs of this predicted box with all GTs
            ious = iou_mat[i]

            # ignore already-matched GTs
            ious = np.where(matched_gt, -1.0, ious)

            gt_idx = np.argmax(ious)
            max_iou = ious[gt_idx]

            if max_iou >= thr:
                TP += 1
                matched_gt[gt_idx] = True

        FP = len(predicted_boxes) - TP
        FN = len(ground_truth_boxes) - matched_gt.sum()

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = 2*precision*recall / (precision+recall) if precision+recall > 0 else 0.0

        results.append(dict(
            iou_threshold = thr,
            TP = int(TP),
            FP = int(FP),
            FN = int(FN),
            precision = precision,
            recall = recall,
            f1_score = f1
        ))

    return pd.DataFrame(results)


def read_test_annotations(annotation_path):
    """
    Read test annotations from a CSV file.
    The CSV file is expected to have columns: x,y,w,h
    return boxes list in XYXY format
    """
    import pandas as pd
    annotations = pd.read_csv(annotation_path)
    boxes = []
   
    for _, row in annotations.iterrows():
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]
        # w+=2
        # h+=2
        # # x2, y2 = x1+w, y1+h
        # x1 = x- w/2
        # y1 = y - h/2
        # x2 = x + w/2
        # y2 = y + h/2
        x1 = x-2
        y1 = y-2
        x2 = x + w
        y2 = y + h
        boxes.append([x1, y1, x2, y2])
    return boxes

def read_detected_annotations(annotation_path):
    """
    Read detected annotations from a CSV file.
    The CSV file is expected to have columns: x1,y1,x2,y2,score
    return boxes list in XYXY format
    """
    import pandas as pd
    annotations = pd.read_csv(annotation_path)
    boxes = []
    scores = []
   
    for _, row in annotations.iterrows():
        x1, y1, x2, y2, score = row["x1"], row["y1"], row["x2"], row["y2"], row["score"]
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
    return boxes, scores

def tile_images_with_labels(image, xxyy_boxes, tile_size=128, overlap=0.2):
    height, width = image.shape[:2]
    stride = int(tile_size * (1 - overlap))

    tiled_images = []
    tiled_boxes = []

    def is_in_box(inner_box, border):
        x1,y1,x2,y2 = inner_box
        x,y,x_end,y_end = border
        return x<=x1<=x_end and x<=x2<=x_end and y<=y1<=y_end and y<=y2<=y_end

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            # check if the tile is smaller than tile_size, if so, adjust the start position
            if x_end - x < tile_size:
                x = max(0, x_end - tile_size)
            if y_end - y < tile_size:
                y = max(0, y_end - tile_size)
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            tile = image[y:y_end, x:x_end]
            tiled_images.append(tile)

            # check all the xxyy_boxes, if it in the image, translate it
            boxes_in_tile = []
            for box in xxyy_boxes:
                boarder_box = (x, y, x_end, y_end)
                if is_in_box(box, boarder_box):
                    x1, y1, x2, y2 = box
                    translated_box = [x1 - x, y1 - y, x2 - x, y2 - y]
                    boxes_in_tile.append(translated_box)
            tiled_boxes.append(boxes_in_tile)
    return tiled_images, tiled_boxes

def get_tp_fp_fn_indices(predicted_boxes, confidence_scores, ground_truth_boxes, iou_threshold=0.5):
    """
    Based on greedy matching: predictions sorted by confidence.
    Returns:
        TP_pred_idx: indices of predicted boxes that matched a GT
        FP_pred_idx: indices of predicted boxes that failed to match
        FN_gt_idx: indices of GT boxes that were never matched
    """

    predicted_boxes = np.array(predicted_boxes)
    ground_truth_boxes = np.array(ground_truth_boxes)
    confidence_scores = np.array(confidence_scores)

    # Sort predictions by confidence descending
    if len(confidence_scores) > 0:
        sort_idx = np.argsort(-confidence_scores)
        predicted_boxes = predicted_boxes[sort_idx]
        confidence_scores = confidence_scores[sort_idx]
    else:
        sort_idx = np.arange(len(predicted_boxes))

    # Precompute IoU matrix
    M = len(predicted_boxes)
    N = len(ground_truth_boxes)

    if M == 0 and N == 0:
        return [], [], []

    if M == 0:
        return [], [], list(range(N))

    if N == 0:
        return [], list(range(M)), []

    # Vectorized IoU computation
    pb = predicted_boxes[:, None, :]  # M×1×4
    gt = ground_truth_boxes[None, :, :]  # 1×N×4

    xA = np.maximum(pb[..., 0], gt[..., 0])
    yA = np.maximum(pb[..., 1], gt[..., 1])
    xB = np.minimum(pb[..., 2], gt[..., 2])
    yB = np.minimum(pb[..., 3], gt[..., 3])

    inter = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)

    area_pred = (pb[..., 2] - pb[..., 0]) * (pb[..., 3] - pb[..., 1])
    area_gt = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])

    iou_mat = inter / np.maximum(area_pred + area_gt - inter, 1e-9)

    # Matching arrays
    matched_gt = np.zeros(N, dtype=bool)

    TP_pred_idx = []
    FP_pred_idx = []

    for p in range(M):
        ious = iou_mat[p]

        # mask out GT boxes already matched
        masked = np.where(matched_gt, -1.0, ious)
        gt_idx = np.argmax(masked)
        best_iou = masked[gt_idx]

        if best_iou >= iou_threshold:
            TP_pred_idx.append(p)
            matched_gt[gt_idx] = True
        else:
            FP_pred_idx.append(p)

    # FN = GT that were never matched
    FN_gt_idx = np.where(~matched_gt)[0].tolist()

    # Convert prediction indexes back to original order
    # (optional — remove this if you prefer sorted order)
    TP_pred_idx = sort_idx[TP_pred_idx].tolist()
    FP_pred_idx = sort_idx[FP_pred_idx].tolist()

    return TP_pred_idx, FP_pred_idx, FN_gt_idx

def draw_tp_fp_fn(
    image, 
    predicted_boxes, 
    confidence_scores, 
    ground_truth_boxes, 
    TP_pred_idx, 
    FP_pred_idx, 
    FN_gt_idx,
    line_thickness=1
):
    """
    Draw TP/FP/FN boxes:
      TP: green + score
      FP: red + score
      FN: blue
    """

    img = image.copy()

    # Color definitions (BGR)
    GREEN = (0, 255, 0)
    RED   = (0, 0, 255)
    BLUE  = (255, 0, 0)

    # ----------- Draw TP -----------
    for p in TP_pred_idx:
        box = predicted_boxes[p]
        score = confidence_scores[p]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, line_thickness)

        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    # ----------- Draw FP -----------
    for p in FP_pred_idx:
        box = predicted_boxes[p]
        score = confidence_scores[p]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, line_thickness)

        label = f"{score:.2f}"
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

    # ----------- Draw FN (GT boxes not matched) -----------
    for g in FN_gt_idx:
        box = ground_truth_boxes[g]

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, line_thickness)

        cv2.putText(img, "FN", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)

    return img