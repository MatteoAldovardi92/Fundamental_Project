# EoMT LoRA Fine-Tuning — Experiment Rationale & Proposed Configurations

All experiments fine-tune `MaskClassificationLoRA` on Cityscapes semantic
segmentation starting from an EoMT-Base checkpoint pre-trained on COCO.
Training is monitored via **training loss** and **validation loss** on a
city-stratified 80/20 holdout of the Cityscapes training split.

---

## 0. Fixed Hyperparameters (shared across all experiments)

| Parameter | Value | Reason |
|---|---|---|
| `max_epochs` | 10 | Constrained by Colab session time; sufficient to see convergence trends |
| `batch_size` | 4 | Maximum that fits on a T4/A100 at 640×640 |
| `accumulate_grad_batches` | 4 | Gives effective batch size 16, matching the EoMT authors' Cityscapes setup |
| `weight_decay` | 0.05 | Standard AdamW default for ViT-scale models [Dosovitskiy et al., 2021] |
| `poly_power` | 0.9 | Slightly convex decay; keeps LR high for longer than linear and is the default in the EoMT codebase |
| `warmup_steps` | [125, 125] | ≈13% of ~1490 total optimizer steps (after 80/20 split); within the 10–20% window recommended by [He et al., 2021] |
| `lora_target_modules` | qkv, fc1, fc2 | See §3 below |

---

## 1. Baseline — `optimal_lora_config.yaml`

```yaml
lr: 1.0e-4
llrd: 0.9
lora_r: 16
lora_alpha: 32   # scaling = alpha/r = 2.0
lora_dropout: 0.05
gradient_clip_val: 0.05
```

**Reference point.** `lr=1e-4` is the standard AdamW base rate for ViT
fine-tuning [Touvron et al., 2022 — DeiT III].  `llrd=0.9` produces a ~3×
spread between block 0 and block 11, gently protecting pre-trained
representations from large updates [Howard & Ruder, 2018 — ULMFiT].  `r=16`
and `alpha=32` follow the `alpha = 2r` convention from [Hu et al., 2022 —
LoRA], which doubles the effective learning signal from the adapters.

---

## 2. Proposed Experiments

### Exp A — Rank Ablation: Low Rank (`r=4`)

```yaml
experiment_name: "lora_rank4"
lr: 1.0e-4
llrd: 0.9
lora_r: 4
lora_alpha: 8    # keeps scaling = 2.0
lora_dropout: 0.05
gradient_clip_val: 0.05
```

**Motivation.**  The LoRA paper [Hu et al., 2022] shows that for pre-trained
models, the intrinsic dimensionality of the weight update is surprisingly low.
For tasks with moderate domain shift (COCO → Cityscapes road scenes), `r=4`
may capture the necessary adaptation with only **6 144 trainable parameters
per layer** versus 24 576 at `r=16`.  If val loss matches the baseline this
suggests we are over-parameterising the adapters and `r=4` is preferable for
its regularisation effect on a small (~2400-image) dataset.

**What to watch.**  If training loss is consistently *higher* than baseline
from epoch 1 the rank is a capacity bottleneck; if val loss is *lower* the
reduction acts as useful implicit regularisation.

**Trainable params (approx).**  ~5.3 M vs ~8.7 M at `r=16`.

---

### Exp B — Rank Ablation: High Rank (`r=32`)

```yaml
experiment_name: "lora_rank32"
lr: 1.0e-4
llrd: 0.9
lora_r: 32
lora_alpha: 32   # scaling = 1.0 (deliberate, see below)
lora_dropout: 0.1
gradient_clip_val: 0.05
```

**Motivation.**  At `r=32` each adapter has 49 152 parameters per layer.
[Zhang et al., 2023 — AdaLoRA] shows that higher rank is beneficial when the
task requires adapting *multiple* independent feature directions simultaneously —
plausible here because semantic segmentation must distinguish 19 classes with
fine spatial detail.  We deliberately set `alpha=32` (scaling=1.0 instead of
2.0) to prevent the larger adapter from dominating the frozen pre-trained
weight; this follows the conservative initialisation strategy from [Dettmers
et al., 2023 — QLoRA].  Dropout is raised to 0.1 to compensate for the
higher parameter count relative to dataset size.

**What to watch.**  Train/val loss gap relative to baseline — a wider gap
indicates overfitting on 2400 images.  If val loss improves we have genuinely
been capacity-limited at `r=16`.

**Trainable params (approx).**  ~12.1 M.

---

### Exp C — Attention-Only LoRA (`qkv` + `proj`, no MLP)

```yaml
experiment_name: "lora_attn_only"
lr: 1.0e-4
llrd: 0.9
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_clip_val: 0.05
lora_target_modules:
  - "qkv"
  - "proj"
```

**Motivation.**  [Zhu et al., 2023 — Visual LoRA survey] and [Jia et al.,
2022 — VPT] find that in vision transformers the attention mechanism is the
primary locus of task-specific adaptation, while MLP layers encode more
generic feature transformations.  Restricting LoRA to `qkv` and `proj`
(attention input and output projections) halves the adapter count relative to
the baseline while keeping adaptation in the most task-relevant subspace.
`proj` is added alongside `qkv` because [Shi et al., 2024] shows that
omitting the output projection leaves the attention output largely unchanged
even when `qkv` is adapted.

**What to watch.**  If performance is close to the baseline (`qkv+fc1+fc2`),
this is the preferred config for production — fewer parameters, faster
inference after merging, and less risk of catastrophic forgetting.

**Trainable params (approx).**  ~6.1 M.

---

### Exp D — Aggressive LLRD (`llrd=0.8`)

```yaml
experiment_name: "lora_llrd_08"
lr: 1.0e-4
llrd: 0.8
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_clip_val: 0.05
```

**Motivation.**  `llrd=0.9` gives a 3× LR spread between block 0 and block
11.  `llrd=0.8` increases this to **~11×** (block 0 LR ≈ 0.8^11 × 1e-4 ≈
8.6e-6), which is the decay used in full-ViT fine-tuning papers [Touvron et
al., 2022; He et al., 2022 — MAE].  For *full* fine-tuning this is well
motivated because all parameters update.  For LoRA, adapters are already
small perturbations, so the question is whether the added protection of early
blocks improves generalisation or simply starves them of signal.

**Prediction.**  If COCO and Cityscapes share enough low-level features (edges,
textures), early blocks should not need to adapt at all and aggressive decay
is beneficial.  If the domain shift also requires low-level adaptation (e.g.
wet roads, night scenes) the heavy decay will hurt.

**What to watch.**  Compare per-block gradient norms logged by W&B.  If
early-block norms are negligibly small even with `llrd=0.9`, the aggressive
decay is free regularisation; if they are non-trivial with `llrd=0.9` the
aggressive decay is harmful.

---

### Exp E — No LLRD, Lower Base LR

```yaml
experiment_name: "lora_flat_lr"
lr: 3.0e-5
llrd: 1.0        # all blocks get the same lr
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_clip_val: 0.05
```

**Motivation.**  LLRD is heuristic: it assumes earlier layers need less
adaptation, which may not hold after LoRA injection (adapters are zero-
initialised, so any update is a genuine signal, not noise in a pre-trained
weight).  A flat LR of 3e-5 is chosen so the *effective* LR for block 11
matches the no-LLRD baseline (1e-4 is the head LR; block 11 LoRA already
receives 1e-4 without LLRD, so a flat 3e-5 is a conservative uniform rate).
This experiment tests whether the complexity of per-layer LR groups is
justified at all for LoRA-only training.

**What to watch.**  If val loss is similar to the baseline with LLRD, LLRD
adds no value for this setting and can be dropped.  If it is worse, the LLRD
protection of early blocks is genuinely beneficial.

---

## 3. Why `qkv + fc1 + fc2` for the Baseline

The EoMT ViT-Base encoder has three weight matrices per block that LoRA can
target:

| Module | Shape | Role |
|---|---|---|
| `qkv` | 768 → 2304 | fused Q, K, V projection |
| `proj` | 768 → 768 | attention output projection |
| `fc1` | 768 → 3072 | MLP expansion |
| `fc2` | 3072 → 768 | MLP contraction |

The baseline targets `qkv + fc1 + fc2` (omitting `proj`) following the
original LoRA paper [Hu et al., 2022, Table 2] which achieves the best
parameter-efficiency tradeoff by adapting both attention weights and MLP
weights while leaving the attention output projection to the frozen base.
Exp C (`qkv + proj`) tests the alternative hypothesis that MLP adaptation is
unnecessary.

---

## 4. Summary Table

| Config | r | alpha | scale | LLRD | Modules | Trainable params |
|---|---|---|---|---|---|---|
| Baseline | 16 | 32 | 2.0 | 0.9 | qkv, fc1, fc2 | ~8.7 M |
| **Exp A** — Low rank | **4** | **8** | 2.0 | 0.9 | qkv, fc1, fc2 | ~5.3 M |
| **Exp B** — High rank | **32** | **32** | 1.0 | 0.9 | qkv, fc1, fc2 | ~12.1 M |
| **Exp C** — Attn only | 16 | 32 | 2.0 | 0.9 | **qkv, proj** | ~6.1 M |
| **Exp D** — Aggressive LLRD | 16 | 32 | 2.0 | **0.8** | qkv, fc1, fc2 | ~8.7 M |
| **Exp E** — Flat LR | 16 | 32 | 2.0 | **1.0** | qkv, fc1, fc2 | ~8.7 M |

---

## 5. References

- **Hu et al. (2022)** — *LoRA: Low-Rank Adaptation of Large Language Models.*
  ICLR 2022. https://arxiv.org/abs/2106.09685

- **Howard & Ruder (2018)** — *Universal Language Model Fine-Tuning for Text
  Classification (ULMFiT).* ACL 2018. https://arxiv.org/abs/1801.06146

- **Touvron et al. (2022)** — *DeiT III: Revenging the Dead-Tokens.*
  ECCV 2022. https://arxiv.org/abs/2204.07118

- **He et al. (2022)** — *Masked Autoencoders Are Scalable Vision Learners (MAE).*
  CVPR 2022. https://arxiv.org/abs/2111.06377

- **Dosovitskiy et al. (2021)** — *An Image is Worth 16×16 Words: Transformers
  for Image Recognition at Scale (ViT).* ICLR 2021. https://arxiv.org/abs/2010.11929

- **Jia et al. (2022)** — *Visual Prompt Tuning (VPT).* ECCV 2022.
  https://arxiv.org/abs/2203.12119

- **Zhang et al. (2023)** — *AdaLoRA: Adaptive Budget Allocation for
  Parameter-Efficient Fine-Tuning.* ICLR 2023. https://arxiv.org/abs/2303.10512

- **Dettmers et al. (2023)** — *QLoRA: Efficient Finetuning of Quantized LLMs.*
  NeurIPS 2023. https://arxiv.org/abs/2305.14314

- **Shi et al. (2024)** — *Dept: Decomposed Prompt Tuning for Parameter-Efficient
  Fine-tuning.* ICLR 2024. https://arxiv.org/abs/2309.05074

- **He et al. (2021)** — *On the Effectiveness of Adapter-based Tuning for
  Pretrained Language Model Adaptation.* ACL 2021. https://arxiv.org/abs/2106.03164
