from ultralytics import YOLO, RTDETR
import torch
import numpy as np
import cv2

from utils import predict_with_tiles, compute_metrics, read_test_annotations

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
    model_path = "runs/detect/train3/weights/best.pt"
    test_image_path = "datasets/test/test_image.tif"
    label_path = "datasets/test/test_image_groundtruth.csv"



    # draw the ground truth boxes on the test image and save it
    src_image = cv2.imread("datasets/test/04_Detection Output.tiff")
    image = cv2.imread(test_image_path)

    print(src_image.shape, image.shape)

    ground_truth_boxes = read_test_annotations(label_path, src_image.shape[:2], image.shape[:2])

    for box in ground_truth_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite("ground_truth_boxes.png", image)


    exit(0)

    model = load_model(model_type='rtdetr', model_path=model_path, device='cpu')
    final_boxes, final_scores, final_class_ids = predict_with_tiles(model, test_image_path, tile_size=640, overlap=0.2, imgsz=640, conf=0.25)

    # save final_boxes to csv
    np.savetxt("predicted_boxes.csv", final_boxes, delimiter=",", fmt="%.2f")
    # save ground_truth_boxes to csv
    np.savetxt("ground_truth_boxes.csv", ground_truth_boxes, delimiter=",", fmt="%.2f")


    metrics = compute_metrics(final_boxes, final_scores, ground_truth_boxes, iou_threshold=0.1)
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")