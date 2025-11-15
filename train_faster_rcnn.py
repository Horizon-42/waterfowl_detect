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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import pandas as pd

class YoloDetectionDataset(Dataset):
    """Dataset wrapper that reads YOLO-format labels and returns torchvision targets."""

    def __init__(self, images_dir, class_count, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = os.path.join(os.path.dirname(images_dir), "labels")
        self.class_count = class_count
        self.transforms = transforms

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = []
        for file_name in sorted(os.listdir(self.images_dir)):
            if os.path.splitext(file_name)[1].lower() not in exts:
                continue
            image_path = os.path.join(self.images_dir, file_name)
            label_path = os.path.join(self.labels_dir, os.path.splitext(file_name)[0] + ".txt")
            self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No image files detected in {self.images_dir}")

    def __len__(self):
        """Return the number of image/label pairs available."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a sample and convert YOLO boxes to absolute XYXY tensors."""
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

        if self.transforms is not None:
            image, target = self.transforms(image, target)

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
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] += v.item()

        if step % print_freq == 0:
            avg = {k: running[k] / step for k in running}
            loss_str = ", ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(f"Epoch {epoch} | Step {step}/{len(data_loader)} | {loss_str}")


@torch.inference_mode()
def evaluate_loss(model, data_loader, device):
    """Compute average loss terms on the validation loader without gradients."""


    model.eval()
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
    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


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
    train_images = resolve_path(dataset_root, config["train"]) if dataset_root else resolve_path(base_dir, config["train"])
    val_key = config.get("val", None)
    val_images = None
    if val_key:
        val_images = resolve_path(dataset_root, val_key) if dataset_root else resolve_path(base_dir, val_key)
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

    loss_rows = []

    train_images, val_images, class_count = load_dataset_paths(args.data_yaml)

    # Only add simple augmentation so the evaluation distribution remains clean.
    train_transforms = RandomHorizontalFlip(probability=args.hflip) if args.hflip > 0 else None
    val_transforms = None

    train_dataset = YoloDetectionDataset(train_images, class_count, transforms=train_transforms)
    val_dataset = None
    if val_images is not None and os.path.isdir(val_images):
        val_dataset = YoloDetectionDataset(val_images, class_count, transforms=val_transforms)

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
    torch.optim.AdamW()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        lr_scheduler.step()

        if val_loader is not None:
            val_metrics = evaluate_loss(model, val_loader, device)
            val_loss = sum(val_metrics.values()) if val_metrics else float("inf")
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()) if val_metrics else "n/a"
            print(f"Validation epoch {epoch}: total_loss={val_loss:.4f} | {metrics_str}")
            
            row = {"epoch": epoch, "val_loss": val_loss}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            loss_rows.append(row)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on YOLO-format datasets")
    parser.add_argument("--data_yaml", type=str, default="faster_rcnn.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Initial learning rate")
    parser.add_argument("--lr_step", type=int, default=15, help="LR scheduler step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader worker processes")
    parser.add_argument("--print_freq", type=int, default=20, help="Logging frequency per epoch")
    parser.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip probability for training")
    parser.add_argument("--output_dir", type=str, default="runs/fasterrcnn", help="Directory to store checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="How often (in epochs) to write checkpoints")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet-pretrained backbone")
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    main(args)
