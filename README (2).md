# Week 6: Malaria parasite detection in thin-blood-smear cell images with ResNet-18

> **Course:** Introduction to Applied AI, Project 2 (individual) | **Notebook:** `week6-malaria-parasite-detection.ipynb` | **Run on:** Kaggle, CPU, seed 42
> **Metric declared before training:** sensitivity (recall of *Parasitized*) ≥ 0.95, always reported with specificity and PPV. Models selected by validation **macro-F1**, not accuracy.

---

**Who benefits:** laboratory technicians and microscopists who screen thin blood smears in clinics with few trained readers. The model is a **triage aid** that tells the reader which cells or slides to look at first; it is not a diagnosis.

**Why it matters (cost or harm):** a missed infected cell can mean untreated malaria; a false alarm wastes reader time or leads to unnecessary treatment. At the operating point used here the per-cell error rates are:

$$\text{miss rate}=1-\text{sens}=1-0.972=0.028\;(\approx 28\text{ missed per }1{,}000\text{ infected cells}),\qquad \text{false-alarm rate}=1-\text{spec}=1-0.936=0.064\;(\approx 64\text{ per }1{,}000\text{ healthy cells})$$

*(One cited statistic on malaria burden, for example from the WHO World Malaria Report, can be added here. ✏️ verify and cite before submitting.)*

---

**Dataset:** *Cell Images for Detecting Malaria* (Kaggle, `iarunava`), <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>; thin-smear cell crops from the NIH / NLM collection. **License:** ✏️ copy the exact licence shown on the Kaggle dataset page.

| Item | Value | Calculation |
|---|---|---|
| Images | **27,558** | 13,779 (Parasitized) + 13,779 (Uninfected) |
| Classes $K$ | **2** | `Parasitized`, `Uninfected` |
| Patient/slide groups $G$ | **200** | $27{,}558\,/\,G = 137.79 \Rightarrow G = 200$ |
| Groups with Parasitized cells | 150 | printed by the notebook |
| Groups with Uninfected cells | 200 | printed by the notebook |
| Patients with healthy cells only | **50** | $200-150$ |
| Image size (300-image sample) | 61 × 82 min, **130 × 133 median**, 229 × 187 max; all RGB | resize factor at the median: $224/130\approx1.72$ |

**Labels by:** expert slide readers, according to the dataset's source publication (Rajaraman et al., *PeerJ* 2018). The 150 infected + 50 healthy patient split matches the 150 / 200 groups found above. ✏️ verify these details against the paper before submitting.

---

**Data audit:**

| Check | Result | How it was computed |
|---|---|---|
| Imbalance ratio | **1.0** | $n_{\max}/n_{\min}=13{,}779/13{,}779$ |
| Images per group | **137.79** | $N/G=27{,}558/200$ |
| Exact duplicates (MD5) | **0** (0 across splits, 0 label conflicts) | identical MD5 hash = identical file |
| Test images with a near-duplicate in train | **0 of 3,335** | fingerprint distance $d<0.05$ |
| Val images with a near-duplicate in train | **0 of 5,105** | same test |
| Closest test→train distance | **0.065** | $\approx 9\times$ the 0.007 distance of a recompressed copy ($0.065/0.007$) |
| Split by | **patient/slide id** (regex `^(C\d+(?:P\d+)?)`) | `StratifiedGroupKFold`, 6 folds, applied twice |
| Shared groups (train/val, train/test, val/test) | **0 / 0 / 0** | set intersections of group ids |

**Near-duplicate fingerprint.** Each image is shrunk to a $32\times32$ grayscale thumbnail $a$ (1,024 pixels), standardised, and compared with its closest training image:

$$z=\frac{a-\mu_a}{\sigma_a},\qquad d(z_1,z_2)=\frac{1}{1024}\sum_{p=1}^{1024}\lvert z_{1,p}-z_{2,p}\rvert,\qquad \text{flag if } d<0.05$$

The four closest pairs ($d=0.065,\,0.067,\,0.068,\,0.070$) were inspected: all are pale, round, featureless Uninfected cells that are *different* cells (different tint and outline), so nothing was removed. A threshold of 0.10 would have flagged about 480 such unrelated cells.

**Split sizes.** With $k=6$ folds the expected shares are val $\approx 1/6=16.7\%$, test $\approx\frac56\cdot\frac16=13.9\%$, train $\approx(\frac56)^2=69.4\%$. Whole patients cannot be divided, so the realised split differs:

| Split | Images | Share of 27,558 | Parasitized | Uninfected | % Parasitized |
|---|---|---|---|---|---|
| train | 19,118 | 69.4 % | 9,586 | 9,532 | 50.1 % |
| val | 5,105 | 18.5 % | 2,853 | 2,252 | 55.9 % |
| test | 3,335 | 12.1 % | 1,340 | 1,995 | **40.2 %** |

Check: $19{,}118+5{,}105+3{,}335=27{,}558$. Because patients are indivisible, the test prevalence is 40.2 % rather than 50 %, which is why PPV is reported at several prevalences below.

---

**Imbalance handling:** weighted cross-entropy (`USE_SAMPLER = False`).

$$w_c=\frac{N}{K\,n_c},\qquad L=-\frac1N\sum_i w_{y_i}\log p(y_i)$$

$$w_{\text{Parasitized}}=\frac{19{,}118}{2\times9{,}586}=\frac{19{,}118}{19{,}172}=\mathbf{0.9972},\qquad w_{\text{Uninfected}}=\frac{19{,}118}{2\times9{,}532}=\frac{19{,}118}{19{,}064}=\mathbf{1.0028}$$

Check: $9{,}586\times0.9972=9{,}559.0$ and $9{,}532\times1.0028=9{,}559.0$, both equal to $N/K=9{,}559$, so each class contributes equally. The training imbalance is only $9{,}586/9{,}532=1.006$, so the weights are about 1 and the weighted loss behaves almost like plain cross-entropy; it is kept so the pipeline stays correct on imbalanced data. Weights come from the **training split only**.

**Augmentation (training only, after the split):** random resized crop (scale 0.8 to 1.0), horizontal and vertical flips, rotation ±180° (a cell has no preferred orientation), and *mild* colour jitter (brightness ±15 %, contrast ±15 %, saturation ±10 %, hue ±0.02 ≈ ±7.2°) because stain colour is both a slide-level nuisance and the evidence itself.

---

**Model and training:** ResNet-18, ImageNet weights loaded from a local Kaggle Model file (no internet), new head $512\to2$.

$$\text{trainable (phase 1)}=512\times2+2=1{,}026\;\;(\approx0.009\,\%\text{ of }11{,}177{,}538),\qquad \text{trainable (phase 2)}\approx 8{,}393{,}728+1{,}026=8{,}394{,}754\;(\approx75\,\%)$$

| Phase | Trainable | Epochs | LR | Best val macro-F1 |
|---|---|---|---|---|
| 1 | head only | 2 | $10^{-3}$ | **0.855** (epochs: 0.834, 0.855) |
| 2 | `layer4` + head | 3 | $10^{-4}$ / $10^{-3}$ | **0.938** (epochs: 0.921, 0.923, 0.938) |

Fine-tuning gain: $0.938-0.855=+0.083$ (relative $+9.7\,\%$). The best epoch is chosen by

$$F1_k=\frac{2P_kR_k}{P_k+R_k},\qquad \text{macro-F1}=\frac1K\sum_{k=1}^{K}F1_k$$

Training time on CPU: $915+922+1{,}169+1{,}179+1{,}181=5{,}366\text{ s}\approx89.4\text{ min}$ (mean epoch ≈ 17.9 min). Validation macro-F1 was still rising at the last epoch, so more epochs would likely help.

---

**Results (test, used once):**

Confusion matrix at the default rule $t=0.5$ (positive = Parasitized): **TP = 1,283, FN = 57, FP = 78, TN = 1,917** (sum 3,335).

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Parasitized | $1{,}283/1{,}361=0.943$ | $1{,}283/1{,}340=\mathbf{0.957}$ | 0.950 |
| Uninfected | $1{,}917/1{,}974=0.971$ | $1{,}917/1{,}995=\mathbf{0.961}$ | 0.966 |

$$\text{Accuracy}=\frac{1{,}283+1{,}917}{3{,}335}=0.960,\qquad \text{Macro-F1}=\frac{0.950+0.966}{2}=\mathbf{0.958},\qquad \text{Weighted-F1}=\frac{0.950\times1{,}340+0.966\times1{,}995}{3{,}335}=0.960$$

The two averages differ by only 0.002 because the test set is nearly balanced. The **weakest class is Parasitized** (F1 0.950, precision 0.943).

**Binary operating point (threshold set on validation, applied once to test).** The threshold is the highest $t$ whose validation true-positive rate reaches the target:

$$t=\max\{\,t:\ \text{TPR}_{\text{val}}(t)\ge0.95\,\}=\mathbf{0.283}$$

| Metric (test, $t=0.283$) | Value | Formula |
|---|---|---|
| Sensitivity | **0.972** | $TP/(TP+FN)$; target 0.95 **reached** (margin +2.2 points) |
| Specificity | **0.936** | $TN/(TN+FP)$ |
| PPV | **0.910** | $TP/(TP+FP)$ |
| AUC | **0.991** | area under the ROC curve |

Trade-off versus $t=0.5$: sensitivity $0.957\to0.972$ (+0.015), specificity $0.961\to0.936$ (−0.025). In counts this is about $1{,}340\times0.028\approx38$ missed cells instead of 57 (≈19 fewer) for $1{,}995\times0.064\approx128$ false alarms instead of 78 (≈50 more). *(Counts at $t=0.283$ are approximated from the printed rates.)*

**PPV by hand at different prevalences** (sens = 0.972, spec = 0.936):

$$\text{PPV}=\frac{\text{sens}\cdot\text{prev}}{\text{sens}\cdot\text{prev}+(1-\text{spec})(1-\text{prev})}$$

| Prevalence of Parasitized | Working | PPV |
|---|---|---|
| 40.2 % (test set, $1{,}340/3{,}335$) | $\dfrac{0.972\times0.402}{0.972\times0.402+0.064\times0.598}=\dfrac{0.3907}{0.4290}$ | **0.911** (notebook prints 0.910 from unrounded inputs) |
| 20 % | $\dfrac{0.1944}{0.1944+0.0512}$ | 0.792 |
| **5 % (assumed screening prevalence)** | $\dfrac{0.972\times0.05}{0.972\times0.05+0.064\times0.95}=\dfrac{0.0486}{0.1094}$ | **0.444** |
| 1 % | $\dfrac{0.00972}{0.00972+0.06336}$ | 0.133 |

Per 1,000 cells at 5 % prevalence: 50 infected × 0.972 = 48.6 caught; 950 healthy × 0.064 = 60.8 false alarms; PPV $=48.6/(48.6+60.8)=0.444$. **Plain language:** sensitivity and specificity do not change, but when infected cells are rare even a 6.4 % false-alarm rate produces more false flags than true ones, so more than half of the positive flags would be wrong. The 5 % value is an assumption and should be justified from local data; this dataset is balanced by design.

---

**Grad-CAM:** for the last block (`layer4`, 512 maps of size $7\times7$ since $224/32=7$) and class score $y^c$:

$$\alpha_k=\frac1Z\sum_i\sum_j\frac{\partial y^c}{\partial A^k_{ij}},\qquad L^c=\text{ReLU}\Big(\sum_k\alpha_kA^k\Big)$$

The map is up-sampled to 224 × 224 and overlaid on six random test cells (3 correct, 3 wrong; seed 0). **Where the model looks:** on correctly classified *Uninfected* cells the heat sits on the **cell centre** (legitimate evidence). On the three **missed Parasitized cells** the heat is on the cell body or near the black segmentation border, and in one case the dark purple parasite spot is "cold", so the model ignored visible evidence.

**Shortcut risk: suspected, not confirmed.** What was checked and what is still open:

- All three correct examples are Uninfected, so the figure does not yet show what drives a correct *Parasitized* prediction.
- Possible shortcuts: slide-level stain tint (pink vs grey-blue) and the black segmentation mask. The 50 healthy-only patients make a slide-style → label correlation possible.
- Proposed tests: (1) Grad-CAM on correct Parasitized cells; (2) mask the black background or crop to the cell interior and re-score; (3) remove stain colour (grayscale or stain normalisation) and re-score; (4) external validation on another hospital's slides. Grad-CAM is coarse ($7\times7$): it detects shortcuts, it does not localise parasites.

---

**Limitations:** data from a single collection site and staining protocol; pre-segmented single-cell crops (a real pipeline needs cell detection first); no parasite species or life-stage labels; cell-level, not slide- or patient-level, evaluation; class mix balanced by design, unlike a real smear; short CPU training (5 epochs, ≈ 89 min); threshold tuned on a validation set with 55.9 % prevalence. **Not for** stand-alone diagnosis, treatment decisions, or use on slides from other microscopes, stains or hospitals without local validation.

---

## The four responsibility questions

| Question | Answer |
|---|---|
| **Who could be harmed, and how?** | A patient whose infected cell is missed (about 2.8 % of infected cells at $t=0.283$) and whose treatment is delayed; a healthy person flagged by a false alarm. At 5 % prevalence $1-0.444=55.6\,\%$ of positive flags would be false. |
| **Where was it NOT validated?** | Other hospitals, microscopes, cameras and stain protocols; smartphone images; thick smears; other parasite species or stages; different age groups; patients in Pakistan; whole-slide use without upstream cell segmentation. |
| **What is the human role?** | Triage only: prioritise cells/slides for a microscopist. Never a final diagnosis; confirm with microscopy, a rapid test or PCR. |
| **What would change my conclusions?** | External validation on local slides with PPV at the local prevalence; Grad-CAM showing heat on the parasite for correct Parasitized predictions; masking and stain-removal tests that do not collapse performance; slide-level evaluation. |

## Reproducibility

Seed 42; `StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)`; Adam; batch size 32 (train), 64 (val/test); $\lceil19{,}118/32\rceil=598$ training batches per epoch; weights saved to `/kaggle/working/week6_model.pt` together with `classes`, `backbone`, `img_size` and `grayscale`.
