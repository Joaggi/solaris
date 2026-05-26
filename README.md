# Solaris: A Foundation Model of the Sun

This repository contains Solaris model code and training utilities for 12-hour
multi-wavelength solar forecasting experiments on processed SDO/AIA imagery.

Paper: [Solaris: A Foundation Model of the Sun](https://arxiv.org/abs/2411.16339v1)

## Current Checkpoints

- Patch-size-8 Solaris-Small: <https://huggingface.co/hrrsmjd/solaris_small_patch8>
- Patch-size-4 Solaris-Small: <https://huggingface.co/hrrsmjd/solaris_small_patch4>

The current patch-size-8 checkpoint was trained with:

- Chronological 80/10/10 split over valid samples, with a 24-hour guard band at split boundaries
- Solaris-Small, patch size 8
- Two history frames (`t-12h`, `t`) predicting all eight pretraining wavelengths at `t+12h`
- AdamW with `weight_decay=0.05`, betas `(0.9, 0.95)`, and no decay on bias, norm, positional/time embedding, or learned normalization scalar parameters
- BF16 autocast forward pass with FP32 loss reduction
- EMA disabled

Full-test raw-scale metrics for the current patch-size-8 checkpoint are included
in the Hugging Face model card.

## Repository Structure

```text
solaris/
  model/                  # Solaris encoder, decoder, Swin, Perceiver, and helper modules
  train.py                # Local training entrypoint
  load_data.py            # Pretraining/downstream PyTorch datasets
  download_data.py        # Dataset download and split CLI
  normalization.py        # Solaris normalization and learned transform
  utils_data.py           # Timestamp, path, metadata, and wavelength helpers
modal_train.py            # Optional Modal training/eval/plot entrypoint
scripts/                  # Dataset processing and upload utilities
```

Generated artifacts such as checkpoints, evaluation JSON/Markdown, plots, HDF5
data, and W&B logs are ignored by Git.

## Local Setup

The local path is the default way to run the code. Use any Python 3.11
environment with enough disk space for the processed HDF5 files and a CUDA GPU
for practical training runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install W&B only if you want experiment tracking:

```bash
python -m pip install wandb==0.18.7
```

## Data Preparation

Download the processed AIA HDF5 files from Hugging Face and build chronological
train/validation/test ID files:

```bash
python -m solaris.download_data \
  --data-dir data/AIA_12hour_512x512 \
  --years 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
  --make-pretrain-splits
```

For a quick check, download fewer years:

```bash
python -m solaris.download_data \
  --data-dir data/AIA_12hour_512x512 \
  --years 2019 2020 \
  --make-pretrain-splits
```

## Training

Run a short local quick check:

```bash
python -m solaris.train \
  --data_path data/AIA_12hour_512x512 \
  --id_dir data/AIA_12hour_512x512/splits \
  --patch_size 8 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --max_steps 100 \
  --run_name quick_check_p8
```

A full Solaris-Small patch-size-8 run uses the same entrypoint with the full data
range and larger batch settings:

```bash
python -m solaris.train \
  --data_path data/AIA_12hour_512x512 \
  --id_dir data/AIA_12hour_512x512/splits \
  --patch_size 8 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --max_steps 7750 \
  --run_name solaris_small_p8
```

The training loss is the Solaris weighted MAE on the normalized intensity scale.
Progress logs also report per-channel RMSE after unscaling predictions back to
the original raw intensity scale.

## Optional Modal Usage

Modal can be useful when you do not have a suitable local GPU. It is optional and
will create cloud resources in your own Modal account. The default training and
plotting functions request an H100 GPU, large ephemeral disk, and a persistent
Modal volume, so you should check Modal pricing before launching long runs.
Persistent volumes can continue to incur storage cost after compute stops.

Install the Modal extras and authenticate:

```bash
python -m pip install -r requirements-modal.txt
modal setup
```

Choose your own app and volume names:

```bash
export SOLARIS_MODAL_APP_NAME=solaris-pretrain
export SOLARIS_MODAL_VOLUME_NAME=solaris-aia-data
```

Prepare the remote dataset and run a short quick check:

```bash
modal run modal_train.py --mode prepare
modal run modal_train.py --mode quick_check --patch-size 8
```

Launch a detached training run only after confirming the expected cost:

```bash
modal run --detach modal_train.py \
  --mode train_detached \
  --steps 7750 \
  --patch-size 8 \
  --no-use-ema
```

For W&B logging on Modal, create a Modal secret containing `WANDB_API_KEY`, set
`SOLARIS_MODAL_WANDB_SECRET_NAME` to that secret name before running
`modal_train.py`, and pass `--use-wandb`.

## Citation

If you use this code or find our work helpful, please cite:

```bibtex
@article{abdulmajid2024solaris,
  title={Solaris: A Foundation Model of the Sun},
  author={Abdul Majid, Harris and Sittoni, Pietro and Tudisco, Francesco},
  journal={arXiv preprint arXiv:2411.16339},
  year={2024}
}
```
