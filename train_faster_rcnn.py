"""Faster R-CNN training entrypoint for YOLO-format datasets.

This script loads YOLO-style label files, converts them into the format expected by
`torchvision`'s Faster R-CNN implementation, and runs a configurable training
loop with optional validation loss tracking and checkpointing.
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2


class YoloDetectionDataset(Dataset):
    """Dataset wrapper that reads YOLO-format labels and returns torchvision targets."""

    def __init__(self, images_dir, class_count, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
        self.class_count = class_count
        self.transforms = transforms

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(
                f"Labels directory not found: {self.labels_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = []
        for file_name in sorted(os.listdir(self.images_dir)):
            if os.path.splitext(file_name)[1].lower() not in exts:
                continue
            image_path = os.path.join(self.images_dir, file_name)
            label_path = os.path.join(
                self.labels_dir, os.path.splitext(file_name)[0] + ".txt")
            self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No image files detected in {self.images_dir}")

    def __len__(self):
        """Return the number of image/label pairs available."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a sample and convert YOLO boxes to absolute XYXY tensors. Supports Albumentations transforms."""
        image_path, label_path = self.samples[idx]
        image = self._load_image(image_path)
        height, width = image.shape[1], image.shape[2]

        boxes = []
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, w, h = map(float, parts)
                    cls = int(cls)
                    if cls < 0 or cls >= self.class_count:
                        continue
                    box = self._yolo_to_xyxy(xc, yc, w, h, width, height)
                    if box is None:
                        continue
                    boxes.append(box)
                    labels.append(cls + 1)

        # Albumentations expects boxes as list of [x_min, y_min, x_max, y_max] and labels as list
        if self.transforms is not None:
            # Albumentations expects HWC numpy image
            np_image = image.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
            # If no boxes, pass empty list
            transformed = self.transforms(
                image=np_image,
                bboxes=boxes,
                labels=labels
            )
            # Convert back to torch tensor (C, H, W)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert boxes and labels to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }

        return image, target

    @staticmethod
    def _yolo_to_xyxy(xc, yc, w, h, width, height):
        """Convert normalized YOLO coordinates into absolute XYXY pixel positions."""
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height

        x1 = max(0.0, min(x1, width))
        y1 = max(0.0, min(y1, height))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _load_image(path):
        """Read image or numpy array and return a contiguous CHW float tensor in [0,1]."""
        with Image.open(path) as img:
            # grayscale images are converted to RGB
            img = img.convert("RGB")
            tensor = F.to_tensor(img)
        return tensor.contiguous()


class RandomHorizontalFlip:
    """Lightweight transform that mirrors images and their boxes with given probability."""

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability and target["boxes"].shape[0] > 0:
            _, _, width = image.shape
            image = F.hflip(image)
            boxes = target["boxes"]
            x_min = width - boxes[:, 2]
            x_max = width - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes
        return image, target


def collate_fn(batch):
    """Custom collate function to keep images and targets in list form."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_model(num_classes, pretrained=True):
    """Instantiate a Faster R-CNN model with the requested number of classes."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20):
    """Run a single training epoch with optional progress logging."""
    model.train()
    running = defaultdict(float)
    for step, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        # Prevent exploding gradients when large tiles or boxes spike the loss.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] += v.item()

        if step % print_freq == 0:
            avg = {k: running[k] / step for k in running}
            loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(
                f"Epoch {epoch} | Step {step}/{len(data_loader)} | {loss_str}")


@torch.inference_mode()
def evaluate_loss(model, data_loader, device):
    """Compute average loss terms on the validation loader without gradients."""
    was_training = model.training
    model.train()
    totals = defaultdict(float)
    count = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
            batch_size = len(images)
            count += batch_size
            for k, v in loss_dict.items():
                totals[k] += v.item() * batch_size
    if not was_training:
        model.eval()
    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


@torch.inference_mode()
def evaluate_metrics(model, data_loader, device, iou_threshold=0.5):
    """
    Compute detection metrics (mAP, precision, recall, F1) on the validation loader.
    This is a simplified implementation, only consider 1 class.
    """
    model.eval()

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

    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            # iterate for every image in the batch
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_num = len(gt_boxes)
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                if gt_num == 0:
                    FP += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0:
                    FN += len(gt_boxes)
                    continue

                # convert gt_boxes to set for matching
                gt_boxes = list(gt_boxes)
                # sort predictions by scores in descending order
                sorted_indices = np.argsort(-pred_scores)
                pred_boxes = [pred_boxes[i] for i in sorted_indices]
                pred_scores = [pred_scores[i] for i in sorted_indices]

                for pb in pred_boxes:
                    if len(gt_boxes) == 0:
                        FP += 1
                        continue
                    ious = [iou(pb, gb) for gb in gt_boxes]
                    # get max iou and its index
                    max_idx = np.argmax(ious)
                    max_iou = ious[max_idx]
                    if max_iou >= iou_threshold:
                        TP += 1
                        gt_boxes.pop(max_idx)
                    else:
                        FP += 1
                FN += len(gt_boxes)

    # calculate precision, recall, F1
    precision = TP/(TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP/(TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0.0
    # mAP calculation is simplified here; for single class, it's equivalent to precision
    mAP = precision

    # For simplicity, we will return dummy values
    metrics = {
        "mAP": mAP,
        "precision": precision,
        "recall": recall,
        "F1": F1
    }
    return metrics


def resolve_path(base_dir, relative_path):
    """Resolve dataset paths regardless of whether they are absolute or relative."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.normpath(os.path.join(base_dir, relative_path))


def load_dataset_paths(data_yaml):
    """Load train/val directories and class count from a YOLO `data.yaml`."""
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(data_yaml))
    dataset_root = config.get("path", "")
    if dataset_root:
        dataset_root = resolve_path(base_dir, dataset_root)
    train_images = resolve_path(
        dataset_root, config["train"]) if dataset_root else resolve_path(base_dir, config["train"])
    val_key = config.get("val", None)
    val_images = None
    if val_key:
        val_images = resolve_path(
            dataset_root, val_key) if dataset_root else resolve_path(base_dir, val_key)
    class_count = int(config["nc"])
    return train_images, val_images, class_count


def save_checkpoint(state, output_dir, filename):
    """Persist model/optimizer state dictionaries to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def main(args):
    """Drive the end-to-end Faster R-CNN training workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # check if output directory exists, if already exists, create new directory
    if os.path.exists(args.output_dir):
        base_output_dir = args.output_dir
        suffix = 1
        while True:
            new_output_dir = f"{base_output_dir}_{suffix}"
            if not os.path.exists(new_output_dir):
                args.output_dir = new_output_dir
                break
            suffix += 1
    # create output directory
    os.makedirs(args.output_dir)

    train_images, val_images, class_count = load_dataset_paths(args.data_yaml)

    # Use Albumentations for data augmentation, could keep box and label mapping consistent.
    train_transforms = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(scale=(0.8, 1.2), shear=(-5, 5),
                 translate_percent=(0.0, 0.1), p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 3), p=0.4),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.NoOp()
        ], p=0.4),
        A.RandomBrightnessContrast(
            brightness_limit=0.02, contrast_limit=0.02, p=0.5),  # small
        A.GaussNoise(p=0.4),
        # custom: gain/offset jitter (implement as lambda if needed)
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    val_transforms = None

    train_dataset = YoloDetectionDataset(
        train_images, class_count, transforms=train_transforms)
    val_dataset = None
    if val_images is not None and os.path.isdir(val_images):
        val_dataset = YoloDetectionDataset(
            val_images, class_count, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        # Keep validation deterministic by disabling shuffling and augmentation.
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    num_classes = class_count + 1
    model = create_model(num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # wait decay, to prevent overfitting, L2 regularization
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # loss and metrics tracking
    loss_rows = []
    metrics = []

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device,
                        epoch, print_freq=args.print_freq)
        lr_scheduler.step()

        if val_loader is not None:
            val_metrics = evaluate_loss(model, val_loader, device)
            val_loss = sum(val_metrics.values()
                           ) if val_metrics else float("inf")
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in val_metrics.items()) if val_metrics else "n/a"
            print(
                f"Validation epoch {epoch}: total_loss={val_loss:.4f} | {metrics_str}")

            row = {"epoch": epoch, "val_loss": val_loss}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            loss_rows.append(row)

            obd_metrics = evaluate_metrics(model, val_loader, device)
            obd_metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in obd_metrics.items()) if obd_metrics else "n/a"
            print(
                f"Validation epoch {epoch} detection metrics: {obd_metrics_str}")
            obd_row = {f"epoch": epoch}
            obd_row.update({f"val_{k}": v for k, v in obd_metrics.items()})
            metrics.append(obd_row)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    args.output_dir,
                    "best_model.pt",
                )

        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                args.output_dir,
                f"checkpoint_epoch_{epoch}.pt",
            )

    save_checkpoint(
        {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        args.output_dir,
        "last_model.pt",
    )

    if loss_rows:
        df = pd.DataFrame(loss_rows)
        csv_path = os.path.join(args.output_dir, "training_log.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved training log: {csv_path}")

    # save metrics
    if metrics:
        df = pd.DataFrame(metrics)
        csv_path = os.path.join(args.output_dir, "detection_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved detection metrics log: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on YOLO-format datasets")
    parser.add_argument("--data_yaml", type=str,
                        default="faster_rcnn.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Initial learning rate")
    parser.add_argument("--lr_step", type=int, default=15,
                        help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float,
                        default=0.1, help="LR scheduler gamma")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader worker processes")
    parser.add_argument("--print_freq", type=int, default=20,
                        help="Logging frequency per epoch")
    parser.add_argument("--hflip", type=float, default=0.5,
                        help="Horizontal flip probability for training")
    parser.add_argument("--output_dir", type=str,
                        default="runs/fasterrcnn", help="Directory to store checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="How often (in epochs) to write checkpoints")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet-pretrained backbone")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main(args)
