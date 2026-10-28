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

Comparison between pixel-level (**ERFNet**) and mask-level (**EoMT**) representations across 5 benchmark datasets:

| Benchmark Dataset | ERFNet (MaxLogit)<br>AUPRC / FPR95 | EoMT (MSP)<br>AUPRC / FPR95 | EoMT (MaxLogit)<br>AUPRC / FPR95 | EoMT (**RbA**)<br>**AUPRC** / FPR95 |
| :--- | :---: | :---: | :---: | :---: |
| **SMIYC RoadAnomaly21** | 38.30% / 59.37% | 17.92% / 89.62% | 28.91% / 75.84% | **55.14%** / 72.42% |
| **SMIYC RoadObstacle21** | 4.63% / 48.46% | 3.55% / 100.0% | 28.50% / 99.99% | **37.52%** / 100.0% |
| **Fishyscapes Lost & Found** | 3.28% / 45.48% | 0.76% / 43.83% | 0.48% / 91.40% | **5.04%** / 55.03% |
| **Fishyscapes Static** | 6.27% / 39.68% | 5.56% / 91.28% | 5.64% / 78.23% | **18.81%** / 70.03% |
| **Road Anomaly** | 15.37% / 72.17% | 10.62% / 94.18% | 14.47% / 85.88% | **16.79%** / 93.91% |

**Key takeaway:** EoMT paired with Residual-based Anomaly (RbA) dramatically outperforms pixel baselines on AUPRC (**55.14% vs 38.30%** on RoadAnomaly21 and **37.52% vs 4.63%** on RoadObstacle21), as object-level queries suppress false positive noise on boundary edges.

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
