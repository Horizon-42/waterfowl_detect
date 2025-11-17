# Waterfowl Detection Project

This project focuses on waterfowl detection in thermal images using deep learning models such as YOLO and RT-DETR. The repository contains scripts for training, evaluation, data augmentation, and utility functions for handling datasets and predictions.

## Project Structure

```
waterfowl_detect/
│
├── Assigment1.md                  # Assignment report and analysis
├── data_set.ipynb                 # Jupyter notebook for dataset exploration/experiments
├── data.yaml                      # YOLO-format dataset configuration
├── YOLO.py                        # YOLO training, tiling, and prediction script
├── RTDETR.py                      # RT-DETR training and evaluation script
├── train_faster_rcnn.py           # Faster R-CNN training script for YOLO-format data
├── test_faster_rcnn.py            # Faster R-CNN evaluation/testing script
├── test_yolo_rtdetr.py            # Testing script for YOLO/RT-DETR models
├── utils.py                       # Utility functions (tiling, metrics, annotation parsing, etc.)
│
├── datasets/                       # Dataset root directory
│   ├── raw_datas/                  
│   └── yolo_tiled_waterfowl_balanced/
│       ├── train/
│       │   ├── images/             # Training images (npy or image files)
│       │   └── labels/             # YOLO-format label files
│       └── val/
│           ├── images/             # Validation images
│           └── labels/             # Validation labels
│
├── runs/                          # Training and evaluation outputs
│   └── detect/
│       ├── train*/                # Training runs (YOLO/RT-DETR/Faster R-CNN)
│       │   ├── args.yaml
│       │   ├── results.csv
│       │   └── weights/
│       └── val*/                  # Validation outputs
├── results/                       # Test resulsts files
├── docs/                          # Documentation and analysis
|   ├── Assigment1_Waterfowl.md 
│   └── Assigment2_Task1_DeepMetricAnalysis.md
│
└── data_augmentations_verify.ipynb # Notebook for verifying data augmentations
```

## Notes
- The project supports both YOLO-format and custom datasets, with scripts for tiling, augmentation, and evaluation.
- See `docs/Assigment1_Waterfowl.md ` and `docs/Assigment2_Task1_DeepMetricAnalysis.md` for detailed analysis and results.

## Getting Started
- Install requirements: `pip install -r requirements.txt`
- Prepare your dataset in YOLO format and update `data.yaml`.
- Use the provided scripts to train, evaluate, and analyze models.

---
For more details, refer to the scripts and documentation in the repository.
