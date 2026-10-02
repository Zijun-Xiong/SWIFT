# Image classification

This experiment fine-tunes CLIP ViT-B/32 on the five datasets reported in the paper: Cars, DTD, GTSRB, RESISC45, and SVHN.

The SWIFT layer is implemented in `goat/peta/src/goat/layer.py`. `goat/train_vit.py` passes `swift_num_blocks` into `GOATConfig`; the paper setting is `N=64`.

## Setup

```bash
cd SWIFT-GitHub
conda env create -f image_classification/environment.yml
conda activate swift-vision
bash image_classification/scripts/install_upstream_extras.sh
python scripts/doctor.py --experiment image_classification
python tests/test_swift_structure.py
python tests/smoke_vision_peft.py
```

This follows the official GOAT setup: Python 3.9, PyTorch 2.4.0 with CUDA 12.1, torchvision 0.19.0, Transformers 4.44.0, PEFT 0.14.0, DeepSpeed 0.15.4, FlashAttention 2.7.3, and the remaining published requirements. The default paper launcher uses four GPUs, but `GPU_IDS=0 TASKS=SVHN` provides a one-GPU functional run after the SVHN download.

Datasets are stored under `data/` by default. GTSRB and SVHN can be downloaded automatically:

```bash
cd image_classification/goat
python download_data.py --data_root ../data
cd ..
```

Cars, DTD, and RESISC45 require manual download or acceptance of their original terms. Their expected layouts are:

```text
data/stanford_cars/devkit/{cars_train_annos.mat,cars_meta.mat}
data/stanford_cars/{cars_train,cars_test,cars_test_annos_withlabels.mat}
data/dtd/{train,val}/<class>/*.jpg
data/resisc45/NWPU-RESISC45/<class>/*.jpg
data/resisc45/resisc45-{train,val,test}.txt
```

After placing them, validate from `image_classification/` with `python goat/download_data.py --data_root data --tasks Cars DTD RESISC45`. Set `DATA_ROOT` to an existing shared dataset directory when appropriate.

## Train and evaluate

```bash
bash scripts/train_swift.sh
```

The script loops over all five paper datasets and evaluates each test split through the training callback. It writes test accuracies to `results/results.csv` and each final SWIFT adapter to `output/<run-name>/`. Defaults use four GPUs (`GPU_IDS=0,1,2,3`), CLIP ViT-B/32, total batch size 512, learning rate `1e-4`, and `N=64`. Use `TASKS="SVHN"` for a quick single-dataset run. Weights & Biases is installed for GOAT environment parity but logging is disabled by default; set `REPORT_TO=wandb` to enable it.

To load a saved adapter, construct the same `CLIPForImageClassification` base model and call `peta.src.load_swift_adapter(model, adapter_directory)` before evaluation.

A smallest real-data launch from `image_classification/` is:

```bash
GPU_IDS=0 TASKS=SVHN TOTAL_BATCH_SIZE=64 PER_DEVICE_BATCH_SIZE=64 \
DATA_ROOT=/data/swift-vision bash scripts/train_swift.sh
```

