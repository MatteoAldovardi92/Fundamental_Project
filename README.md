# EoMT: Cross-Task Evaluation, LoRA Fine-Tuning & Anomaly Detection

> **Project for the course *Fundamentals of AI* (A.Y. 2025/2026) — Politecnico di Torino**  
> **Author:** Matteo Aldovardi  
> **Based on:** *Your ViT is Secretly an Image Segmentation Model* (CVPR 2025 Highlight) — [arXiv:2503.19108](https://arxiv.org/abs/2503.19108)

---

## 1. Project Overview & Objectives

This project investigates the **Encoder-only Mask Transformer (EoMT)** architecture on dense scene understanding and safety-critical anomaly detection for autonomous driving. 

Unlike traditional segmentation frameworks (e.g., Mask2Former) that require specialized pixel decoders or transformer adapters, EoMT repurposes a plain Vision Transformer (ViT) to jointly encode image patch tokens and segmentation query tokens in self-attention, removing decoder overhead entirely.

The project is structured into three main milestones:
1. **Cross-Task & Zero-Shot Evaluation (Step 4):** Evaluate pre-trained EoMT checkpoints across different datasets (COCO $\to$ Cityscapes) and test bicubic positional embedding interpolation to handle varying image resolutions.
2. **LoRA Fine-Tuning (Step 5):** Adapt the 86M parameter ViT-B backbone to Cityscapes semantic segmentation using Low-Rank Adaptation (LoRA) across three experimental configurations.
3. **Out-of-Distribution & Anomaly Segmentation (Steps 7 & 8):** Compare mask-based representations (EoMT) against a real-time pixel-based baseline (**ERFNet**) on safety-critical road obstacle benchmarks using multiple uncertainty and anomaly scoring functions.

---

## 2. Techniques & Evaluation Metrics

### Architectures
- **EoMT (ViT-B backbone, DINOv2 initialization):** Encoder-only mask classification model. Generates class logits and binary mask embeddings directly from query tokens.
- **ERFNet:** Efficient residual factorized ConvNet used as a real-time pixel-level baseline.

### Adaptation & Fine-Tuning
- **Bicubic Positional Embedding Interpolation:** Dynamically resamples 2D positional embeddings when image input dimensions change (e.g., $640 \times 640$ COCO crops to Cityscapes resolutions).
- **LoRA (Low-Rank Adaptation):** Decomposes weight updates in attention projections ($W = W_0 + \frac{\alpha}{r}BA$, with $r=8, \alpha=16$) combined with Layer-wise Learning Rate Decay (LLRD).

### Anomaly Scoring Functions
For detecting out-of-distribution (OOD) obstacles:
- **MSP (Maximum Softmax Probability):** $S_{\text{MSP}}(\mathbf{x}) = 1 - \max_c P(y=c \mid \mathbf{x})$
- **MaxLogit:** $S_{\text{ML}}(\mathbf{x}) = -\max_c z_c(\mathbf{x})$
- **Max Entropy:** $S_{\text{Ent}}(\mathbf{x}) = -\sum_c P(c \mid \mathbf{x}) \log P(c \mid \mathbf{x})$
- **RbA (Residual-based Anomaly):** Mask-level scoring that flags candidate segments that do not confidently match any in-distribution class.

### Metrics
- **mIoU (Mean Intersection over Union):** Primary metric for in-distribution Cityscapes semantic segmentation.
- **AUPRC (Area Under Precision-Recall Curve):** Primary metric for anomaly detection due to extreme class imbalance (anomalies represent $<1\%$ of pixels).
- **FPR@TPR95:** False Positive Rate at 95% True Positive Rate.

---

## 3. Results Scoreboard

### A. LoRA Fine-Tuning on Cityscapes (Validation Set)

Three experiments with increasing numbers of trainable LoRA blocks:

| Experiment | Configuration | Trainable Parameters | Validation mIoU | Best Checkpoint |
| :--- | :--- | :---: | :---: | :--- |
| **Exp 1** | Prediction Head only | Head MLPs | **72.01%** | `val_loss_total=2.138.ckpt` |
| **Exp 2** | Head + Blocks 9, 10, 11 | Head + 3 Decoder blocks | **73.31%** | `val_loss_total=2.084.ckpt` |
| **Exp 3 (Best)** | Head + All 12 ViT Blocks + LLRD | Full Backbone LoRA | **75.39%** | `val_loss_total=2.009.ckpt` |

*Exp 3 achieves $+3.38\%$ mIoU over head-only tuning, confirming that adapting early transformer attention layers improves low-level edge precision.*

---

### B. Road Anomaly & OOD Detection Scoreboard

Evaluation across 5 safety-critical anomaly benchmarks (`resultsAnomalyDetection/`). Comparison between pixel-level (**ERFNet**) and mask-based **EoMT** (COCO zero-shot, Cityscapes pre-trained, and **Fine-tuned with LoRA**):

| Benchmark Dataset | ERFNet (Pixel)<br>MaxLogit | EoMT (COCO)<br>Best Method | EoMT (Cityscapes)<br>Best Method | EoMT (Fine-tuned)<br>MSP / MaxLogit | EoMT (Fine-tuned)<br>**Top Score (Best Method)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SMIYC RoadAnomaly21** | 38.30% / 59.37% | 55.14% / 72.42% *(RbA)* | 68.19% / 26.55% *(MaxLogit)* | 83.46% / 20.45% *(MSP)* | **84.68% / 18.10%** *(Max Entropy)* |
| **SMIYC RoadObstacle21** | 4.63% / 48.46% | 37.52% / 100.0% *(RbA)* | 81.36% / 1.81% *(Max Entropy)* | 77.47% / 2.15% *(MSP)* | **88.84% / 0.30%** *(Max Entropy)* |
| **Fishyscapes Lost & Found** | 3.28% / 45.48% | 5.04% / 55.03% *(RbA)* | 22.67% / 42.83% *(MaxLogit)* | 21.07% / 30.98% *(MSP)* | **62.32% / 34.71%** *(RbA, $T=1.5$)* |
| **Fishyscapes Static** | 6.27% / 39.68% | 18.81% / 70.03% *(RbA)* | 47.82% / 57.14% *(MaxLogit)* | 52.36% / **3.14%** *(MaxLogit)* | **81.04% / 20.21%** *(RbA)* |
| **Road Anomaly** | 15.37% / 72.17% | 38.10% / 77.55% *(RbA)* | 70.55% / 31.19% *(MaxLogit)* | 77.26% / 34.80% *(MSP, $T=1.5$)* | **78.91% / 39.67%** *(Max Entropy)* |

*Format: **AUPRC ↑ / FPR@TPR95 ↓**.*

#### Key Observations from `resultsAnomalyDetection/`:
1. **Massive Fine-Tuning Gain:** LoRA fine-tuning on Cityscapes dramatically improves anomaly detection across all benchmarks (e.g. SMIYC RA-21 jumps from **17.92%** on COCO zero-shot to **84.68% AUPRC** on Fine-tuned EoMT).
2. **Superiority over Pixel Baselines:** The fine-tuned EoMT outperforms ERFNet by a wide margin on all five datasets (e.g., **88.84% vs 4.63% AUPRC** and **0.30% vs 48.46% FPR95** on SMIYC RoadObstacle21).
3. **Temperature Scaling ($T = 1.5$):** Grid search on Road Anomaly (`temperature_search_results.csv`) identifies $T = 1.5$ as optimal, boosting calibrated MSP and pushing Fishyscapes Lost & Found RbA from **58.85%** to **62.32% AUPRC**.
4. **Extreme Imbalance Handling:** As documented in `dataset_anomaly_statistics.csv`, benchmarks like SMIYC RO-21 (mean anomaly: $0.72\%$) and FS Lost & Found (mean anomaly: $0.28\%$, with 91/100 images having $<1\%$ anomaly pixels) present extreme class sparsity where mask representations effectively prevent boundary false alarms.

---

## 4. Repository Structure

```
Fundamental_Project/
├── Step4.ipynb                   # Step 4: Cross-task evaluation & zero-shot transfer
├── Step5.ipynb                   # Step 5: LoRA fine-tuning (Exp 1, Exp 2, Exp 3)
├── Step7-8.ipynb                 # Steps 7–8: Anomaly detection benchmarks
├── requirements.txt              # Environment dependencies
│
├── eomt/                         # Core EoMT library (models, configs, training)
├── eval/                         # Evaluation scripts (IoU, ERFNet, anomaly scorers)
│   ├── iouEval.py                # Per-class IoU calculator
│   ├── evalAnomaly.py            # ERFNet anomaly runner
│   ├── eomt_anomaly_eval.py      # EoMT anomaly runner
│   └── erfnet.py                 # ERFNet model architecture
│
├── resultsAnomalyDetection/      # Raw CSV results for all benchmarks
├── saved_visualizations/         # Class mIoU plot and prediction masks
├── posthoc_metrics/              # Post-hoc evaluation utilities
└── coco-classes-mapping-master/  # COCO class mapping utilities
```

---

## 5. References

- Kerssies et al., *Your ViT is Secretly an Image Segmentation Model*, CVPR 2025.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Romera et al., *ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation*, IEEE T-ITS 2018.
- Chan et al., *SegmentMeIfYouCan: A Benchmark for Detecting Unexpected Obstacles on the Road*, NeurIPS 2021.
