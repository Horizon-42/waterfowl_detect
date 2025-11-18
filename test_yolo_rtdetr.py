from ultralytics import YOLO, RTDETR
import torch
import numpy as np
import cv2
import json
import os
import pandas as pd

from utils import predict_with_tiles, compute_metrics, read_test_annotations, compute_metrics_multi_iou_fast, read_detected_annotations
from utils import get_tp_fp_fn_indices, draw_tp_fp_fn
def load_model(model_type, model_path, num_classes=None, device='cpu'):
    if model_type == 'yolo':
        model = YOLO(model_path)
    elif model_type == 'rtdetr':
        model = RTDETR(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.to(device)
    model.eval()
    return model

def compute_metrics(train_idx:int, image_path:str, label_path:str):
    # draw the ground truth boxes on the test image and save it
    image = cv2.imread(image_path)
    ground_truth_boxes = read_test_annotations(label_path)

    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)


    model = load_model(model_type='yolo', model_path=model_path, device='cuda')
    final_boxes, final_scores, final_class_ids = predict_with_tiles(model, image_path, tile_size=128,
                                                                     overlap=0.1, imgsz=512, conf=0.25)
    
    # save final boxes as csv
    with open(f'detected_boxes{train_idx}.csv', 'w') as f:
        f.write('x1,y1,x2,y2,score,class_id\n')
        for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
            x1, y1, x2, y2 = map(int, box)
            f.write(f'{x1},{y1},{x2},{y2},{score},{class_id}\n')

    metrics:pd.DataFrame = compute_metrics_multi_iou_fast(final_boxes, final_scores, ground_truth_boxes, iou_thresholds=np.arange(0.25, 0.96, 0.05).round(2).tolist())

    metrics.to_csv(f'metrics_{train_idx}.csv', index=False)
    print(metrics)
    # compute the mean precision
    mean_precision = metrics['precision'].mean()
    print(f"Mean Precision: {mean_precision:.4f}")

def visualize_detections(image_path, detected_boxes_path:str, gt_boxes_path:str, output_path:str):
    """
    Visualize detections by drawing TP, FP, FN boxes on the image.
    """
    image = cv2.imread(image_path)
    detected_boxes, scores = read_detected_annotations(detected_boxes_path)
    gt_boxes = read_test_annotations(gt_boxes_path)

    tp_ids, fp_ids, fn_ids = get_tp_fp_fn_indices(detected_boxes,scores, gt_boxes, iou_threshold=0.5)
    image = draw_tp_fp_fn(image, detected_boxes, scores, gt_boxes, tp_ids, fp_ids, fn_ids)

    cv2.imwrite(output_path, image)
    print(f"Detections visualized and saved to {output_path}")

if __name__ == "__main__":
    train_idx = 10
    model_path = f"runs/detect/train{train_idx}/weights/best.pt"
    # test_image_path = "datasets/test/val.tif"
    # label_path = "datasets/test/val.csv"
    compute_metrics(train_idx, "datasets/test/test_image.tif", "datasets/test/birds1.csv")



    # compute metrics
    # compute_metrics(train_idx, test_image_path, label_path)

    # visualize detections
    # detected_boxes_path = f"results/for_val_images/detected_boxes10.csv"
    # output_path = f"results/for_val_images/result_val_{train_idx}.png"
    # visualize_detections(test_image_path, detected_boxes_path, label_path, output_path)

    # draw ground truth boxes on test image
    image = cv2.imread("datasets/test/test_image.tif")
    ground_truth_boxes = read_test_annotations("datasets/test/birds1.csv")
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite("results/for_test_images/ground_truth.png", image)



    
