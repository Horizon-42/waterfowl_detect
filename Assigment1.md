# Waterfowl detection on thermal images.
## Training Pipline
### Dataset Setting
- Training with both positive and negative samples
- Train/Valid ration is 0.8/0.2
- Testing on practical orthomosaic images.
**The dataset the link provide has a wrong groundtruth for the proactical test image, we need switch to version 3.**

### Data Agument
- Random Flip
- Copy Paste
- Mosaic
### Model
We use RTDETR and YOLO.
### Metrics
Since it's binary classify, we use **Precision**, **Recall** and **F1 Score** with IOU >=50%.

# Experiment

|Model|Image Size|Augment|Epochs|Precision iou50|Recall|F1|
|---|---|---|---|---|---|---|
|rtdetr-l.pt|512|No|0|0.8981|0.3750|0.5290|
|rtdetr-l.pt|640|No|0|0.8509|0.4130|0.5561|
||||||
|yolo11l.pt|640|No|350|0.9502|0.3732|0.5360|
|yolo11m.pt|1024|No|100|0.9636|0.3904|0.5557|
|yolo11m.pt|1024|No|1000|0.9720|0.4048|0.5716|
||||||
|yolo11x.pt|640|No|350|0.9487|0.3297|0.4893|
|yolo11x.pt|640|Yes|350|0.9438|0.3918|0.5537|

## Analysis 
What are the strengths and weaknesses of thermal imagery for this task?
### The strengths
Comparing to **RGB**, it captures creatures with body heat, distinguish them from low temperature background objects like grass.
### The Weaknesses
