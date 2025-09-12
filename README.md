# 🚗 Car Brand Classification (Algerian Used Cars)
Deep learning model to classify car brands from 3,000+ images of used cars (BMW, Mercedes, etc.). Uses transfer learning with pre-trained CNNs, image preprocessing, and data augmentation. Includes Grad-CAM explainability and evaluation with accuracy and confusion matrix. Code and demo included.

heck yes—here’s a polished, plug-and-play **README** outline you can drop into your repo. It’s “filled enough” to be useful on day one, with placeholders you’ll replace after you run your scripts (metrics, screenshots, Medium link, etc.).

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Medium](https://img.shields.io/badge/Blog-Medium-black.svg)](#-blog-post)

A deep learning pipeline to classify **car brands from images** (e.g., BMW, Mercedes).
Includes: **data prep**, **stratified splits**, **augmentation**, **EfficientNetV2-B0 fine-tuning**, **evaluation**, and **explainability** (Grad-CAM & saliency).

> 🔗 **Dataset:** *Algerian Used Cars* on Kaggle (add link)
> 🔗 **Medium article:** *(add link)*

---

# 📚 Table of Contents

* [Overview](#-overview)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Quickstart](#-quickstart)
* [Workflow](#-workflow)
* [Training](#-training)
* [Evaluation](#-evaluation)
* [Explainability](#-explainability)
* [Results & Insights](#-results--insights)
* [Reproducibility](#-reproducibility)
* [FAQ / Troubleshooting](#-faq--troubleshooting)
* [Contributing](#-contributing)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---

# 🔎 Overview

This project trains a **CNN classifier** to predict the **brand** of a car from an image. It targets marketplace use-cases: **auto-tagging**, **search filters**, and **moderation**.

**Highlights**

* Transfer learning with **EfficientNetV2-B0** (two-stage fine-tuning).
* **Stratified** train/val/test split to keep class ratios stable.
* **Data augmentation** (flip/rotation) and **\[0,1] normalization**.
* **Early stopping** + **learning rate scheduling**.
* **Grad-CAM** and **saliency** visualizations to inspect what the model attends to.

---

# 🗂 Dataset

* **Source:** Kaggle – *Algerian Used Cars* (insert link)
* **Format:** Folder-per-brand (e.g., `dataset/BMW/*.jpg`).
* **Size:** *(fill from your report, e.g., 3,236 images across 20 brands)*
* **Common formats:** *(e.g., JPEG/PNG)*
* **Typical resolutions:** *(fill with min/median/max from summary)*

> ⚠️ **Respect Data Terms:** Review Kaggle license/terms before redistribution.

**Quick preview & summary**

```bash
# Option A: auto-download from Kaggle
python scripts/analyze_kaggle_algerian_cars_py37.py --auto-download --out reports

# Option B: local path
python scripts/analyze_kaggle_algerian_cars_py37.py --data /path/to/algerian-used-cars --out reports
```

Embed your generated assets:

```markdown
<img src="reports/class_distribution.png" width="520" />
<img src="reports/sample_grid.png" width="520" />
```

---

# 🧱 Project Structure

```
car-brand-classification/
├─ scripts/
│  ├─ analyze_kaggle_algerian_cars_py37.py   # dataset scan, charts, report
├─ src/
│  ├─ data/                                  # (optional) helpers for IO/splits
│  ├─ models/                                # model definitions / loading
│  ├─ train.py                               # fine-tune EfficientNetV2-B0
│  ├─ eval.py                                # evaluation & confusion matrix
│  ├─ explain.py                             # Grad-CAM / saliency
├─ notebooks/                                # (optional) EDA / experiments
├─ reports/
│  ├─ summary.txt
│  ├─ summary.json
│  ├─ images_metadata.csv
│  ├─ class_distribution.png
│  ├─ sample_grid.png
│  ├─ confusion_matrix.png
│  ├─ confusion_matrix_normalized.png
│  ├─ per_class_metrics.png
│  ├─ gradcam_examples.png
├─ assets/                                   # README images, logos, GIFs
├─ requirements.txt
├─ README.md
├─ LICENSE
```

---

# ⚡ Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Kaggle (optional, for auto-download)
# place kaggle.json in ~/.kaggle/ and accept dataset terms

# 3) Dataset summary (creates /reports with charts)
python scripts/analyze_kaggle_algerian_cars_py37.py --auto-download --out reports
# or: --data /path/to/algerian-used-cars
```

**Requirements (sample)**

```
tensorflow>=2.12
scikit-learn
pillow
matplotlib
pandas
numpy
```

---

# 🔄 Workflow

1. **Scan & EDA:** Folder tree, class distribution, resolutions (`scripts/analyze_*`).
2. **Split:** Stratified **train/val/test**.
3. **Input pipeline:** Resize to **224×224**, normalize to **\[0,1]**, augment (flip/rotation).
4. **Fine-tune:** EfficientNetV2-B0 with **two-stage** training, **early stopping**, **ReduceLROnPlateau**.
5. **Evaluate:** Accuracy, Top-3, Precision/Recall/F1, confusion matrices.
6. **Explain:** Grad-CAM & saliency overlays.

---

# 🏋️ Training

Minimal example (adapt if you used a notebook):

```python
# src/train.py (excerpt)
model = build_efficientnetv2_b0(num_classes=len(classes))
history_head = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop, reduce_on_plateau])
unfreeze_top_layers(model, pct=0.3)
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop, reduce_on_plateau])
model.save("models/effnetv2b0_best.h5")
```

Command:

```bash
python src/train.py --data /path/to/dataset --out models/ --epochs 20 --batch 32
```

---

# 📈 Evaluation

Computes accuracy, Top-3, macro/weighted Precision/Recall/F1, and confusion matrices.

```bash
python src/eval.py --model models/effnetv2b0_best.h5 --data /path/to/dataset --out reports
```

Embed the outputs:

```markdown
**Overall metrics**
- Accuracy: **XX.XX%**
- Top-3 Accuracy: **YY.YY%**
- Macro F1: **ZZ.ZZ%**

<img src="reports/confusion_matrix.png" width="520" />
<img src="reports/confusion_matrix_normalized.png" width="520" />

<img src="reports/per_class_metrics.png" width="520" />
```

---

# 🧠 Explainability

Run **Grad-CAM** and **saliency** on test samples:

```bash
python src/explain.py --model models/effnetv2b0_best.h5 --data /path/to/dataset --out reports --n 12
```

Embed:

```markdown
<img src="reports/gradcam_examples.png" width="720" />
```

**Interpretation guide**

* Grad-CAM should highlight **badges/grilles/headlights**.
* If heat focuses on **background** signs/plates → consider tighter crops, targeted data collection, or more fine-tuning.

---

# 🧪 Results & Insights

> Replace the placeholders with your real numbers/observations.

**Summary (Test Set)**

| Metric               |    Value |
| -------------------- | -------: |
| Top-1 Accuracy       | `xx.xx%` |
| Top-3 Accuracy       | `yy.yy%` |
| Macro F1             | `zz.zz%` |
| Weighted F1          | `aa.aa%` |
| P95 Latency (server) |  `bb ms` |

**Per-Class Highlights**

* ✅ Best classes: *Brand A*, *Brand B* (F1 > 0.95)
* ⚠️ Weak classes: *Brand X*, *Brand Y* (confused with …)

**Key Takeaways**

* Model attends to **front grille + logo** in correct predictions (Grad-CAM).
* Biggest confusions: *Brand M ↔ Brand N* (similar fascia); gather more data / increase augmentation.
* Thresholding at `p ≥ 0.90` gives **\~98% precision** for auto-tagging; route lower-confidence to review.

---

# 🔁 Reproducibility

* **Seed:** 42 for splits/shuffles.
* **Split:** Train 70% / Val 15% / Test 15% (stratified).
* **Input:** 224×224, normalized to \[0,1]; training augment: horizontal flip + rotation(±10%).
* **Backbone:** EfficientNetV2-B0 (ImageNet weights).
* **Two-stage:** head warm-up → unfreeze top \~30% layers (BN layers kept frozen).

**Exact commands**

```bash
# 1) Dataset report
python scripts/analyze_kaggle_algerian_cars_py37.py --data /path/to/dataset --out reports

# 2) Train
python src/train.py --data /path/to/dataset --out models --epochs 20 --batch 32

# 3) Evaluate
python src/eval.py --model models/effnetv2b0_best.h5 --data /path/to/dataset --out reports

# 4) Explainability
python src/explain.py --model models/effnetv2b0_best.h5 --data /path/to/dataset --out reports --n 12
```

---
# ❓ FAQ / Troubleshooting

* **GPU OOM:** lower `BATCH_SIZE`, or use `mixed_float16`.
* **`Gradients are None` in Grad-CAM:** ensure the chosen layer is a conv layer (e.g., `top_conv`) and inputs are `float32`.
* **`'int' object has no attribute 'numpy'`**: use the fixed `_to_py_int` helper (see `explain.py`).

---

# 🤝 Contributing

PRs welcome!

* Open an issue for bugs/ideas.
* Keep PRs focused; include before/after metrics if changing training code.

---

# 📜 License

This project is released under the **MIT License** (see `LICENSE`).
Dataset licensing follows Kaggle’s terms—**do not** redistribute the raw dataset in this repo.

---

# 🙏 Acknowledgments

* Kaggle: *Algerian Used Cars* dataset (add link/credit).
* Keras/TensorFlow team & community examples on transfer learning and Grad-CAM.

---

# 📝 Blog Post

**Medium:** https://medium.com/@sohamrajjain0007/teaching-ai-to-recognize-car-brands-a-deep-learning-journey-6286c907a45d
Summarize: challenge, approach, key results, and favorite Grad-CAM visuals.

---

## Tips for embedding charts & screenshots on GitHub

* Put images in `reports/` or `assets/`.
* Use relative paths; optionally control width:

```markdown
<img src="reports/sample_grid.png" width="520" alt="Sample images by brand" />
```

* Prefer PNG for plots; JPG/WebP for photos to keep repo size down.


