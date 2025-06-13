# Classification Models

This document provides instructions for setting up data and training classification models using real, synthetic, and combined datasets.

## 1. Data Setup

### Synthetic Data
Prepare the synthetic dataset using the following commands:

```bash
mkdir -p datasets/Cardiac-synthetic/
cp -r samples/lvdm_cardiacnet/jpg datasets/Cardiac-synthetic/jpg
cp samples/lvdm_cardiacnet/FileList.csv datasets/Cardiac-synthetic/FileList.csv

python scripts/update_split_filelist.py \
    --csv datasets/Cardiac-synthetic/FileList.csv \
    --train 457 \
    --val 97 \
    --test 101 

python scripts/update_synt_file.py \
    --input datasets/Cardiac-synthetic/FileList.csv \
    --output datasets/Cardiac-synthetic/FileList.csv
```

## 2. Train Classification Models

### Real Data Only
Train models using real data with the following commands:

```bash
# ASD CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --data-dir datasets/CardiacNet/jpg \
    --csv-path datasets/CardiacNet/FileList.csv \
    --sampling_rate 2  \
    --frame-sampling uniform \
    --frames-per-clip 32 \
    --selected_classes 0,1 \
    --backbone r3d_18 \
    --pretrained \
    --epochs 45  \
    --batch-size 8 \
    --output-dir ./experiments/classification/real \
    --experiment-name r3d_18_asd_cv \
    --use-cross-val \
    --num-folds 5

# PAH CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --data-dir datasets/CardiacNet/jpg \
    --csv-path datasets/CardiacNet/FileList.csv \
    --sampling_rate 2 \
    --frame-sampling uniform \
    --frames-per-clip 32 \
    --selected_classes 2,3  \
    --backbone r3d_18 \
    --pretrained \
    --epochs 45 \
    --batch-size 8 \
    --output-dir ./experiments/classification/real \
    --experiment-name r3d_18_pah \
    --use-cross-val \
    --num-folds 5
```

### Synthetic Data Only
Train models using synthetic data with the following commands:

```bash
# ASD CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --sampling_rate 2 \
    --frame-sampling uniform \
    --frames-per-clip 32 \
    --selected_classes 0,1 \
    --backbone r3d_18 \
    --pretrained \
    --epochs 45 \
    --batch-size 8 \
    --output-dir ./experiments/classification/synth \
    --experiment-name r3d_18_asd \
    --use-cross-val \
    --num-folds 5

# PAH CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --sampling_rate 2 \
    --frame-sampling uniform \
    --frames-per-clip 32 \
    --selected_classes 2,3 \
    --backbone r3d_18 \
    --pretrained \
    --epochs 45 \
    --batch-size 8 \
    --output-dir ./experiments/classification/synth \
    --experiment-name r3d_18_pah \
    --use-cross-val \
    --num-folds 5
```

### Combined Real and Synthetic Data
Train models using both real and synthetic data with the following commands:

```bash
# Here we use the original validation and test sets

# ASD CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --synthetic-data-dir datasets/CardiacNet/jpg \
    --synthetic-csv-path datasets/CardiacNet/FileList.csv \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --sampling_rate 2  \
    --frame-sampling uniform  \
    --frames-per-clip 32 \
    --selected_classes 0,1  \
    --use-synthetic-for VAL TEST  \
    --pretrained \
    --backbone r3d_18  \
    --epochs 45  \
    --batch-size 8 \
    --output-dir ./experiments/classification/real_synth \
    --experiment-name r3d_18_asdv1 \
    --use-cross-val  \
    --num-folds 5

# PAH CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --synthetic-data-dir datasets/CardiacNet/jpg \
    --synthetic-csv-path datasets/CardiacNet/FileList.csv \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --sampling_rate 2  \
    --frame-sampling uniform  \
    --frames-per-clip 32 \
    --selected_classes 2,3 \
    --use-synthetic-for VAL TEST  \
    --pretrained \
    --backbone r3d_18 \
    --epochs 45  \
    --batch-size 8 \
    --output-dir ./experiments/classification/real_synth \
    --experiment-name r3d_18_pahv1  \
    --use-cross-val  \
    --num-folds 5
```

```bash
# Here we use only the original test sets
# ASD CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --synthetic-data-dir datasets/CardiacNet/jpg \
    --synthetic-csv-path datasets/CardiacNet/FileList.csv \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --sampling_rate 2  \
    --frame-sampling uniform  \
    --frames-per-clip 32 \
    --selected_classes 0,1  \
    --use-synthetic-for TEST  \
    --pretrained \
    --backbone r3d_18  \
    --epochs 45  \
    --batch-size 8 \
    --output-dir ./experiments/classification/real_synth \
    --experiment-name r3d_18_asdv2 \
    --use-cross-val  \
    --num-folds 5

# PAH CLASSES
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.train \
    --synthetic-data-dir datasets/CardiacNet/jpg \
    --synthetic-csv-path datasets/CardiacNet/FileList.csv \
    --csv-path datasets/Cardiac-synthetic/FileList.csv \
    --data-dir datasets/Cardiac-synthetic/jpg \
    --sampling_rate 2  \
    --frame-sampling uniform  \
    --frames-per-clip 32 \
    --selected_classes 2,3 \
    --use-synthetic-for TEST  \
    --pretrained \
    --backbone r3d_18 \
    --epochs 45  \
    --batch-size 8 \
    --output-dir ./experiments/classification/real_synth \
    --experiment-name r3d_18_pahv2  \
    --use-cross-val  \
    --num-folds 5
```

###  RELABELLING OF SYNTH DATA
It may be neccessary to relabel the synthetic dataset in order to achieve excellent classification results.

```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.classification.predict_and_relabel \
--checkpoint experiments/classification/real/r3d_18_asd_cv/fold_2/best_model.pth  \
--csv-path Echodream/datasets/Cardiac-synthetic/FileList.csv  \
--data-dir Echodream/datasets/Cardiac-synthetic/jpg    \
--output-csv ./experiments/classification/synth/output_predictions_asd.csv  \   
--batch-size 32     \
--frames-per-clip 32 \
--sampling-rate 2  \
--splits ALL  \
--backbone r3d_18 

CUDA_VISIBLE_DEVICES='0' python -m echo.classification.predict_and_relabel   \
--checkpoint experiments/classification/real/r3d_18_pah_cv/fold_1/best_model.pth  \
--csv-path Echodream/datasets/Cardiac-synthetic/FileList.csv   \
--data-dir Echodream/datasets/Cardiac-synthetic/jpg    \
--output-csv ./experiments/classification/synth/output_predictions_pah.csv    \
--batch-size 32     \
--frames-per-clip 32 \
--sampling-rate 2  \
--splits ALL  \
--backbone r3d_18

python -m  echo.classification.merge_by_prob \
--csv_a experiments/classification/synth/output_predictions_asd.csv \
--csv_b experiments/classification/synth/output_predictions_pah.csv \
--output datasets/Cardiac-synthetic/FileList_relabeled.csv

```