# 🩺 Skin Disease Detection with Multi-Modal Deep Learning
**Dermoscopy + Clinical Metadata | CNNs, Convolutional Autoencoders, and Model Interpretability (Grad-CAM, SHAP)**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

Accurate skin disease diagnosis is difficult due to the **subtle visual differences** between lesion types and the absence of contextual patient information in many automated systems. This project implements a **multi-modal deep learning** framework that combines **dermoscopic images** with **structured clinical metadata** (age, sex, lesion location) to improve classification performance and provide **explainable predictions**.

---

## ✨ What’s Inside

- **Three modeling strategies**
  1. **Fixed feature extraction + metadata fusion**  
     Pretrained CNN (e.g., ResNet/MobileNet/VGG) as frozen feature extractor → fuse with metadata → classify via MLP / RandomForest.
  2. **End-to-end fine-tuned CNN + metadata fusion**  
     Joint optimization of image backbone with a small metadata branch; late fusion before classifier head.
  3. **Convolutional autoencoder (CAE) + metadata fusion**  
     Unsupervised image embeddings from a CAE → fuse with metadata → classify via MLP.

- **Interpretability**
  - **Grad-CAM** on image backbone to visualize salient lesion regions.
  - **SHAP** on metadata branch / tabular pipelines to quantify per-feature contributions.
  - **Permutation Feature Importance** to assess metadata significance post-training.

- **Training aids**
  - Class imbalance mitigation: **SMOTE** + **class weights**.
  - **ANOVA F-test** for metadata feature selection (optional).
  - Reproducible splits, metrics, and experiment logging.

---

## 📊 Headline Results

| Approach                                   | Backbone      | Metadata | Metric     | Score  |
|--------------------------------------------|---------------|----------|------------|--------|
| Fixed CNN features + fusion (MLP/RF)       | ResNet/MobileNet | ✔️     | Accuracy   | 0.74–0.76 |
| **Fine-tuned CNN + metadata (best CNN)**   | **MobileNet** | **✔️**  | **Accuracy** | **0.7659** |
| **CAE embeddings + metadata (best overall)** | **CAE + MLP** | **✔️**  | **F1-score** | **0.77** |

**Takeaway:** Incorporating clinical metadata **improves performance across all models**. A thoughtfully designed, **lightweight unsupervised** image encoder with metadata fusion can **outperform heavier CNNs**.

## 🛠️ Techniques & Tools

- **Languages**: Python  
- **Frameworks**: PyTorch, Torchvision  
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, SHAP, Scikit-learn, Imbalanced-learn (SMOTE)  
- **Methods**:  
  - Fine-tuned CNNs (ResNet, MobileNet, VGG16)  
  - Autoencoders  
  - Metadata fusion with MLP & Random Forest  
  - Feature selection (ANOVA F-test)  
  - Class imbalance handling (SMOTE, class weights)  


## 🚀 Applications

- **Early diagnosis** of skin diseases such as melanoma.  
- **AI-assisted dermatology** to support clinicians with both visual and contextual cues.  
- **Trustworthy medical AI** through explainable predictions.  


## 📊 Example Outputs

- **Heatmaps (Grad-CAM)** → Highlight lesion areas contributing to CNN decisions.  
- **SHAP plots** → Show how age, sex, and lesion location impact model predictions.  
- **Performance metrics** → Accuracy, F1-score comparisons across approaches.  
