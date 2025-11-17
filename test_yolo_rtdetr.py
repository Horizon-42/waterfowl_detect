from ultralytics import YOLO, RTDETR
import torch
import numpy as np
import cv2
import json
import os
import pandas as pd

from utils import predict_with_tiles, compute_metrics, read_test_annotations, compute_metrics_multi_iou_fast

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

if __name__ == "__main__":
    train_idx = 10
    model_path = f"runs/detect/train{train_idx}/weights/best.pt"
    test_image_path = "datasets/test/test_image.tif"
    label_path = "datasets/test/birds2.csv"



    # draw the ground truth boxes on the test image and save it
    image = cv2.imread(test_image_path)
    ground_truth_boxes = read_test_annotations(label_path)

    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)


    model = load_model(model_type='yolo', model_path=model_path, device='cuda')
    final_boxes, final_scores, final_class_ids = predict_with_tiles(model, test_image_path, tile_size=128,
                                                                     overlap=0.1, imgsz=512, conf=0.25)
    
    # save final boxes as csv
    with open(f'detected_boxes{train_idx}.csv', 'w') as f:
        f.write('x1,y1,x2,y2,score,class_id\n')
        for box, score, class_id in zip(final_boxes, final_scores, final_class_ids):
            x1, y1, x2, y2 = map(int, box)
            f.write(f'{x1},{y1},{x2},{y2},{score},{class_id}\n')

    metrics:pd.DataFrame = compute_metrics_multi_iou_fast(final_boxes, final_scores, ground_truth_boxes, iou_thresholds=np.arange(0.25, 0.91, 0.05).round(2).tolist())

    metrics.to_csv(f'metrics_{train_idx}.csv', index=False)
    print(metrics)
