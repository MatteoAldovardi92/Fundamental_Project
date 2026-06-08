# EoMT

This is almost the original repository of the authors of EoMT if something is not clear refer to the [original repo](https://github.com/tue-mps/eomt). You will have to use the code in this folder and adapt it with the eval folder to be able to evaluate and train a EoMT model if needed. You can find a EoMT model trained on Cityscapes dataset with the [config file](eomt/configs/dinov2/cityscapes/semantic) at this [link](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Requirements Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation for training

You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch (don't do it, it will be impossible to do it in Colab due to resource contraints):

```bash
python3 main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on Citiscapes segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.

✅ Make sure the total batch size is `devices × batch_size = 16`
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

---

## Project Fine-Tuning (Step 5)

LoRA fine-tuning on Cityscapes is handled by `training/mask_classification_lora.py` via `MaskClassificationLoRA`, which extends `MaskClassificationSemantic`. Three experiments are defined in `configs/experiments/`:

| Config | Trainable components | `gradient_clip_val` |
|---|---|---|
| `exp1_lora_head_only.yaml` | Decoder heads only | 1.0 |
| `exp2_lora_decoder_blocks.yaml` | Heads + LoRA blocks 9–11 | 1.0 |
| `exp3_lora_all_blocks.yaml` | Heads + LoRA all 12 blocks | 0.5 |

All experiments share: `lr=1e-4`, `llrd=0.9`, `weight_decay=0.05`, `warmup_steps=[200,200]`, `poly_power=0.9`, `lora_r=8`, `lora_alpha=16`, `batch_size=4`, `accumulate_grad_batches=4` (effective batch 16), `max_epochs=25`.

Training is monitored via WandB (`eomt-cityscapes-finetuning` project). Logged metrics:
- `losses/train_loss_total`, `train_loss_mask`, `train_loss_dice`, `train_cross_entropy` (step + epoch)
- `losses/val_loss_total`, `val_loss_mask`, `val_loss_dice`, `val_cross_entropy` (epoch)

Checkpoints saved to `checkpoints/<experiment_name>/` — top-2 by `val_loss_total` + `last.ckpt`.
