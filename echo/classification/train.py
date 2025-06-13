import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
from itertools import cycle
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")

from echo.classification.model import get_video_classifier
from echo.classification.data import create_video_dataloaders, VideoDataset
import imageio


def train_epoch(model, dataloader, criterion, optimizer, accelerator, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    debug = False
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if debug == True and batch_idx == 0 and accelerator.is_local_main_process:
            debug_dir = "debug_batches/"
            os.makedirs(debug_dir, exist_ok=True)
            for i, video in enumerate(inputs):
                video_path = (
                    debug_dir
                    + f"class_{targets[i].item()}_batch_{batch_idx}_video_{i}.mp4"
                )
                video_np = video.cpu().numpy()
                # Permute dimensions to (T, H, W, C)
                video_np = np.transpose(video_np, (0, 2, 3, 1))
                # Scale to 0-255 and convert to uint8
                video_np = (video_np * 255).astype(np.uint8)
                with imageio.get_writer(
                    video_path,
                    fps=32,
                ) as writer:
                    for i in range(video_np.shape[0]):
                        writer.append_data(video_np[i])

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 and accelerator.is_local_main_process:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | "
                f"Loss: {running_loss / (batch_idx + 1):.4f} | "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    # Convert metrics to tensors before gathering
    loss_tensor = torch.tensor(
        running_loss / len(dataloader), device=accelerator.device
    )
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    loss_tensor = accelerator.gather(loss_tensor).mean()
    acc_tensor = accelerator.gather(acc_tensor).mean()

    return loss_tensor.item(), acc_tensor.item()


def validate(model, dataloader, criterion, accelerator):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Collect predictions and targets for AUC-ROC
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert metrics to tensors before gathering
    loss_tensor = torch.tensor(
        running_loss / len(dataloader), device=accelerator.device
    )
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    loss_tensor = accelerator.gather(loss_tensor).mean()
    acc_tensor = accelerator.gather(acc_tensor).mean()

    # Calculate AUC-ROC (only on main process)
    if accelerator.is_local_main_process:
        auc_roc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
        print(
            f"Validation Loss: {loss_tensor.item():.4f} | Validation Acc: {acc_tensor.item():.2f}% | AUC-ROC: {auc_roc:.4f}"
        )
    else:
        auc_roc = None

    return loss_tensor.item(), acc_tensor.item(), auc_roc


def test(model, dataloader, accelerator):
    """Test the model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            # Move to CPU for collection
            pred_cpu = accelerator.gather(predicted).cpu().numpy()
            targets_cpu = accelerator.gather(targets).cpu().numpy()

            all_preds.extend(pred_cpu)
            all_targets.extend(targets_cpu)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Convert metric to tensor before gathering
    acc_tensor = torch.tensor(100.0 * correct / total, device=accelerator.device)

    # Gather and reduce metrics from all processes
    acc_tensor = accelerator.gather(acc_tensor).mean()

    # Calculate AUC-ROC (only on main process)
    if accelerator.is_local_main_process:
        auc_roc = roc_auc_score(all_targets, all_preds, multi_class="ovr")
        print(f"Test Acc: {acc_tensor.item():.2f}% | AUC-ROC: {auc_roc:.4f}")
    else:
        auc_roc = None

    return acc_tensor.item(), all_preds, all_targets, auc_roc


def calculate_and_plot_metrics(all_preds, all_targets, classes, output_dir):
    """
    Calculate and plot metrics for model evaluation

    Args:
        all_preds: numpy array of model predictions
        all_targets: numpy array of true labels
        classes: list of class names
        output_dir: directory to save plots
    """
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Convert to numpy arrays if not already
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(metrics_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. Calculate metrics
    accuracy = np.mean(all_preds == all_targets)
    precision = precision_score(
        all_targets, all_preds, average="macro", zero_division=0
    )
    recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    # Save metrics to file
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 3. ROC curve and AUC (one vs rest for multiclass)
    n_classes = len(classes)

    # Check if we need to compute ROC curves (need probabilistic outputs)
    if n_classes > 2:  # Multiclass case
        # For demonstration, we'll create a dummy probabilistic output
        # In a real scenario, you would need to capture model.predict_proba() results

        # Binarize the labels for one-vs-rest ROC
        y_bin = label_binarize(all_targets, classes=range(n_classes))

        plt.figure(figsize=(10, 8))

        # For each class, calculate class-specific metrics
        class_metrics = {}
        for i in range(n_classes):
            # Create binary labels for this class
            y_true_bin = (all_targets == i).astype(np.int32)
            y_pred_bin = (all_preds == i).astype(np.int32)

            # Calculate AUC using the binary predictions
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

            # Store class metrics
            class_metrics[classes[i]] = {
                "precision": float(
                    precision_score(y_true_bin, y_pred_bin, zero_division=0)
                ),
                "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1_score": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
                "auc": float(roc_auc),
            }

        # Save class-specific metrics
        with open(metrics_dir / "class_metrics.json", "w") as f:
            json.dump(class_metrics, f, indent=4)

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(metrics_dir / "roc_curves.png", dpi=300)
        plt.close()

    # 4. Class distribution
    plt.figure(figsize=(12, 6))

    # True distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x=all_targets, palette="viridis")
    plt.title("True Class Distribution")
    plt.xlabel("Class")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")

    # Predicted distribution
    plt.subplot(1, 2, 2)
    sns.countplot(x=all_preds, palette="viridis")
    plt.title("Predicted Class Distribution")
    plt.xlabel("Class")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(metrics_dir / "class_distribution.png", dpi=300)
    plt.close()

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Video Classification Training")

    # Data paths
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file with video metadata",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the video frames",
    )
    parser.add_argument(
        "--synthetic-csv-path",
        type=str,
        default=None,
        help="Path to CSV file with synthetic video metadata",
    )
    parser.add_argument(
        "--synthetic-data-dir",
        type=str,
        default=None,
        help="Directory containing synthetic video frames",
    )
    parser.add_argument(
        "--use-synthetic-for",
        type=str,
        nargs="+",
        default=[],
        choices=["TRAIN", "VAL", "TEST"],
        help="Which splits should use synthetic data",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=1,
        help="Sampling rate for frames (1 means all frames, 2 means every second frame, etc.)",
    )
    parser.add_argument(
        "--selected_classes",
        type=str,
        nargs="+",
        default=None,
        help="List of selected classes to include in the dataset",
    )

    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "r3d_18",
            "r2plus1d_18",
        ],
        help="Model backbone to use",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained weights"
    )
    parser.add_argument(
        "--pool-type",
        type=str,
        default="avg",
        choices=["avg", "max", "attention"],
        help="Type of temporal pooling",
    )

    # Cross-validation options
    parser.add_argument(
        "--use-cross-val",
        action="store_true",
        help="Use cross-validation instead of fixed validation set",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=45, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Initial learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=64,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--frame-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "random"],
        help="Method for sampling frames",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/classification/real",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name of experiment"
    )

    # New arguments for accelerate
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Enable pin_memory for dataloaders (disable if CUDA errors occur)",
    )

    return parser.parse_args()


def train_and_evaluate_cv_fold(
    fold_idx, train_dataset, val_dataset, test_dataset, args, output_dir, accelerator
):
    """
    Train and evaluate a single fold in the cross-validation process

    Args:
        fold_idx: Current fold index
        train_dataset: Training dataset for this fold
        val_dataset: Validation dataset for this fold
        test_dataset: Test dataset
        args: Command line arguments
        output_dir: Output directory
        accelerator: Accelerator instance

    Returns:
        val_acc: Validation accuracy for this fold
        test_acc: Test accuracy for this fold
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    if accelerator.is_local_main_process:
        fold_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(fold_dir / "logs"))
    else:
        writer = None

    if accelerator.is_local_main_process:
        print(f"\n---- Training Fold {fold_idx} ----")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Create model
    num_classes = len(train_dataset.classes)
    model_kwargs = {"num_classes": num_classes}

    if args.backbone in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["pool_type"] = args.pool_type
    elif args.backbone in ["r3d_18", "r2plus1d_18"]:
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["dropout_prob"] = 0.5

    model = get_video_classifier(
        num_classes=model_kwargs.pop("num_classes"),
        backbone=args.backbone,
        **model_kwargs,
    )

    criterion = nn.CrossEntropyLoss(
        weight=train_dataset.class_weights.to(accelerator.device)
    )
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    test_loader = accelerator.prepare(test_loader)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, epoch
        )

        val_loss, val_acc, _ = validate(model, val_loader, criterion, accelerator)

        # Update learning rate
        scheduler.step()

        # Log metrics
        if accelerator.is_local_main_process and writer is not None:
            writer.add_scalar(f"Fold_{fold_idx}/Loss/train", train_loss, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Accuracy/train", train_acc, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Loss/val", val_loss, epoch)
            writer.add_scalar(f"Fold_{fold_idx}/Accuracy/val", val_acc, epoch)
            writer.add_scalar(
                f"Fold_{fold_idx}/LearningRate", optimizer.param_groups[0]["lr"], epoch
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if accelerator.is_local_main_process:
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "classes": train_dataset.classes,
                    },
                    fold_dir / "best_model.pth",
                )

    # Test the model using the best weights
    accelerator.wait_for_everyone()

    checkpoint = torch.load(
        fold_dir / "best_model.pth", map_location=accelerator.device
    )
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, all_preds, all_targets, _ = test(model, test_loader, accelerator)

    if accelerator.is_local_main_process:
        if writer is not None:
            writer.add_scalar(f"Fold_{fold_idx}/Accuracy/test", test_acc, 0)
            writer.close()

        np.savez(
            fold_dir / "test_results.npz",
            predictions=all_preds,
            targets=all_targets,
            accuracy=test_acc,
        )

        classes = train_dataset.classes
        metrics = calculate_and_plot_metrics(all_preds, all_targets, classes, fold_dir)
        print(
            f"Fold {fold_idx} metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}"
        )

    return best_val_acc, test_acc


def run_cross_validation(args, output_dir, accelerator):
    """
    Run k-fold cross-validation

    Args:
        args: Command line arguments
        output_dir: Output directory
        accelerator: Accelerator instance
    """
    # Load dataset metadata
    df = pd.read_csv(args.csv_path)

    # Combine train and validation data for cross-validation
    train_val_df = df[df["Split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)

    # Set up k-fold cross validation
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    # Get test dataset
    from echo.classification.data import get_video_transforms

    test_transform = get_video_transforms("test")
    test_dataset = VideoDataset(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        split="TEST",
        transform=test_transform,
        frames_per_clip=args.frames_per_clip,
        frame_sampling=args.frame_sampling,
        sampling_rate=args.sampling_rate,
        synthetic_csv_path=args.synthetic_csv_path,
        synthetic_data_dir=args.synthetic_data_dir,
        use_synthetic_for_split=args.use_synthetic_for,
        selected_classes=args.selected_classes,
    )

    # Setup for storing results
    fold_val_results = []
    fold_test_results = []

    # Run k-fold cross validation
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        # Create train and validation splits for this fold
        fold_train_df = train_val_df.iloc[train_idx].copy()
        fold_val_df = train_val_df.iloc[val_idx].copy()

        # Set appropriate split values
        fold_train_df["Split"] = "TRAIN"
        fold_val_df["Split"] = "VAL"

        # Save fold datasets temporarily
        fold_csv_path = output_dir / f"fold_{fold_idx}_split.csv"
        combined_df = pd.concat([fold_train_df, fold_val_df, df[df["Split"] == "TEST"]])
        combined_df.to_csv(fold_csv_path, index=False)

        # Create datasets for this fold
        train_transform = get_video_transforms("train")
        val_transform = get_video_transforms("val")

        train_dataset = VideoDataset(
            csv_path=fold_csv_path,
            data_dir=args.data_dir,
            split="TRAIN",
            transform=train_transform,
            frames_per_clip=args.frames_per_clip,
            frame_sampling=args.frame_sampling,
            sampling_rate=args.sampling_rate,
            synthetic_csv_path=args.synthetic_csv_path,
            synthetic_data_dir=args.synthetic_data_dir,
            use_synthetic_for_split=args.use_synthetic_for,
            selected_classes=args.selected_classes,
        )

        val_dataset = VideoDataset(
            csv_path=fold_csv_path,
            data_dir=args.data_dir,
            split="VAL",
            transform=val_transform,
            frames_per_clip=args.frames_per_clip,
            frame_sampling=args.frame_sampling,
            sampling_rate=args.sampling_rate,
            synthetic_csv_path=args.synthetic_csv_path,
            synthetic_data_dir=args.synthetic_data_dir,
            use_synthetic_for_split=args.use_synthetic_for,
            selected_classes=args.selected_classes,
        )

        # Train and evaluate this fold
        val_acc, test_acc = train_and_evaluate_cv_fold(
            fold_idx,
            train_dataset,
            val_dataset,
            test_dataset,
            args,
            output_dir,
            accelerator,
        )

        fold_val_results.append(val_acc)
        fold_test_results.append(test_acc)

        # Clean up temporary CSV
        if accelerator.is_local_main_process:
            fold_csv_path.unlink()

    # Summarize cross-validation results
    if accelerator.is_local_main_process:
        print("\n---- Cross-Validation Results ----")
        for fold_idx in range(args.num_folds):
            print(
                f"Fold {fold_idx}: Val accuracy = {fold_val_results[fold_idx]:.2f}%, Test accuracy = {fold_test_results[fold_idx]:.2f}%"
            )

        print(
            f"\nMean validation accuracy: {np.mean(fold_val_results):.2f}% ± {np.std(fold_val_results):.2f}%"
        )
        print(
            f"Mean test accuracy: {np.mean(fold_test_results):.2f}% ± {np.std(fold_test_results):.2f}%"
        )

        # Save cross-validation summary
        cv_results = {
            "fold_val_accuracies": fold_val_results,
            "fold_test_accuracies": fold_test_results,
            "mean_val_accuracy": float(np.mean(fold_val_results)),
            "std_val_accuracy": float(np.std(fold_val_results)),
            "mean_test_accuracy": float(np.mean(fold_test_results)),
            "std_test_accuracy": float(np.std(fold_test_results)),
        }

        with open(output_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=4)


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Set random seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Set up output directory
    if args.experiment_name is None:
        args.experiment_name = f"{args.backbone}_{int(time.time())}"

    output_dir = Path(args.output_dir) / args.experiment_name

    # Only create directories and write files from the main process
    if accelerator.is_local_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save args
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
        # Set up tensorboard
        if not args.use_cross_val:
            writer = SummaryWriter(log_dir=str(output_dir / "logs"))
        else:
            writer = None
    else:
        writer = None

    # Print info about distributed setup
    if accelerator.is_local_main_process:
        print(f"Using device: {accelerator.device}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Using cross-validation: {args.use_cross_val}")
        if args.use_cross_val:
            print(f"Number of folds: {args.num_folds}")

    # Execute cross-validation if enabled
    if args.use_cross_val:
        import pandas as pd

        run_cross_validation(args, output_dir, accelerator)
        return

    # Otherwise, proceed with standard training using fixed train/val/test sets
    # Set up dataloaders
    if accelerator.is_local_main_process:
        print("Creating dataloaders...")

    dataloader_config = {
        "csv_path": args.csv_path,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "frames_per_clip": args.frames_per_clip,
        "frame_sampling": args.frame_sampling,
        "sampling_rate": args.sampling_rate,
        "selected_classes": args.selected_classes,
        "synthetic_csv_path": args.synthetic_csv_path,
        "synthetic_data_dir": args.synthetic_data_dir,
        "use_synthetic_for": args.use_synthetic_for,
        "pin_memory": args.pin_memory,  # Add pin_memory control
    }

    # Add debug message for pin_memory setting
    if accelerator.is_local_main_process:
        print(f"Using pin_memory: {args.pin_memory}")

    dataloaders = create_video_dataloaders(dataloader_config)

    # Get number of classes
    num_classes = len(dataloaders["train"].dataset.classes)
    if accelerator.is_local_main_process:
        print(f"Number of classes: {num_classes}")

    # Create model
    if accelerator.is_local_main_process:
        print(f"Creating model: {args.backbone}")

    # Prepare model kwargs based on model type
    model_kwargs = {"num_classes": num_classes}

    if args.backbone in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        # model_kwargs['model_name'] = args.backbone
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["pool_type"] = args.pool_type
    elif args.backbone in ["r3d_18", "r2plus1d_18"]:
        # model_kwargs['model_name'] = args.backbone
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["dropout_prob"] = 0.5
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    # Pass the backbone as a parameter, not as part of kwargs
    # This ensures the correct code path in get_video_classifier is taken
    model = get_video_classifier(
        num_classes=model_kwargs.pop("num_classes"),
        backbone=args.backbone,
        **model_kwargs,
    )
    print(f"Class weights: {dataloaders['train'].dataset.class_weights}")
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=dataloaders["train"].dataset.class_weights.to(accelerator.device)
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare for distributed training with accelerate
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, dataloaders["train"], dataloaders["val"]
    )
    test_dataloader = accelerator.prepare(dataloaders["test"])

    # Train model
    if accelerator.is_local_main_process:
        print("Starting training...")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_dataloader, criterion, optimizer, accelerator, epoch
        )

        val_loss, val_acc, _ = validate(model, val_dataloader, criterion, accelerator)

        # Update learning rate
        scheduler.step()

        # Log metrics (only on main process)
        if accelerator.is_local_main_process and writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Save best model (only on main process)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if accelerator.is_local_main_process:
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "classes": dataloaders["train"].dataset.classes,
                    },
                    output_dir / "best_model.pth",
                )
                print(f"Model saved at epoch {epoch} with val acc: {val_acc:.2f}%")

        # Save latest model (only on main process)
        if accelerator.is_local_main_process:
            # Unwrap model before saving
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "classes": dataloaders["train"].dataset.classes,
                },
                output_dir / "latest_model.pth",
            )

    # Make sure all processes are synced before testing
    accelerator.wait_for_everyone()

    # Test best model
    if accelerator.is_local_main_process:
        print("Loading best model for testing...")

    checkpoint = torch.load(
        output_dir / "best_model.pth", map_location=accelerator.device
    )
    # Unwrap the model for loading weights
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, all_preds, all_targets, test_auc_roc = test(
        model, test_dataloader, accelerator
    )

    # Save test results (only on main process)
    if accelerator.is_local_main_process:
        if writer is not None:
            writer.add_scalar("Accuracy/test", test_acc, 0)
            writer.add_scalar("AUC-ROC/test", test_auc_roc, 0)

        np.savez(
            output_dir / "test_results.npz",
            predictions=all_preds,
            targets=all_targets,
            accuracy=test_acc,
            auc_roc=test_auc_roc,
        )

        # Calculate and plot metrics
        classes = dataloaders["train"].dataset.classes
        metrics = calculate_and_plot_metrics(
            all_preds, all_targets, classes, output_dir
        )
        metrics["auc_roc"] = test_auc_roc  # Add AUC-ROC to the metrics dictionary

        # Save updated metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics calculated and saved to {output_dir / 'metrics'}")
        print(
            f"Overall metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}"
        )

        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%, Test AUC-ROC: {test_auc_roc:.4f}")

        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
