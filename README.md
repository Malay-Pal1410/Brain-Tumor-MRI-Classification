# 🧠 Brain Tumor MRI Image Classification

An end-to-end deep learning project for classifying brain MRI images into four categories using Convolutional Neural Networks and transfer learning.

The project compares multiple deep-learning architectures, selects the best-performing model, provides Grad-CAM explainability, and deploys the final model through an interactive Streamlit application.

> **Important:** This project is developed for academic and research purposes only. It is not intended to provide medical diagnosis, treatment recommendations, or clinical tumor localization.

---

## 1. Project Overview

Brain tumor classification from MRI images is an important application of deep learning and medical image analysis.

This project develops a multi-class image classification system capable of classifying brain MRI images into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

Three deep-learning approaches were implemented and evaluated:

1. Custom CNN
2. MobileNetV2
3. EfficientNetB0

After evaluation on the test dataset, MobileNetV2 achieved the highest test accuracy and was selected as the final model for the deployed application.

Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated to provide visual explanations of the regions that contributed strongly to the model's classification decision.

---

## 2. Project Objectives

The main objectives of this project are:

- Build an automated brain MRI image classification pipeline.
- Perform four-class brain MRI classification.
- Develop a baseline Custom CNN model.
- Apply transfer learning using MobileNetV2.
- Apply transfer learning using EfficientNetB0.
- Compare the performance of all trained models.
- Select the best-performing model based on test results.
- Generate class probabilities and prediction confidence.
- Integrate Grad-CAM for model explainability.
- Develop an interactive Streamlit web application.
- Maintain a reproducible machine-learning workflow.

---

## 3. Classification Classes

The model predicts one of the following four classes:

| Class Index | Class |
|---|---|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

The class ordering used by the trained model is therefore:

```python
[
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]
```

---

## 4. Dataset

The dataset is organized into three subsets:

```text
dataset/
├── train/
├── valid/
└── test/
```

Each subset contains four class directories.

```text
train/
├── glioma/
├── meningioma/
├── no_tumor/
└── pituitary/
```

The same class structure is followed for validation and testing.

### Dataset Summary

| Dataset | Images |
|---|---:|
| Training | 1695 |
| Validation | 502 |
| Testing | 246 |

All images are resized to:

```text
224 × 224 pixels
```

with three RGB channels.

---

## 5. Data Preprocessing

The data pipeline was implemented using TensorFlow/Keras.

Important preprocessing settings include:

```text
Image size : 224 × 224
Batch size: 32
Label mode: categorical
Random seed: 42
```

Training data is shuffled while validation and testing data are kept in deterministic order.

The pipeline also uses TensorFlow `AUTOTUNE` prefetching to improve data-loading performance.

---

## 6. Data Augmentation

Data augmentation is used during training to improve model generalization.

The augmentation pipeline includes:

- Horizontal flipping
- Rotation
- Zoom
- Translation
- Contrast adjustment

The implemented augmentation settings include approximately:

```text
Rotation    : ±0.08
Zoom        : ±0.10
Translation : 0.05
Contrast    : 0.10
```

Augmentation is applied only where appropriate during training.

---

## 7. Models Implemented

### 7.1 Custom CNN

A custom Convolutional Neural Network was developed as the baseline model.

The network learns hierarchical visual features directly from the MRI images using convolutional operations followed by classification layers.

The Custom CNN provides a baseline for comparison with pretrained transfer-learning architectures.

---

### 7.2 MobileNetV2

MobileNetV2 was implemented using transfer learning with ImageNet pretrained weights.

Architecture:

```text
Input MRI
   ↓
Data Augmentation
   ↓
MobileNetV2 Preprocessing
   ↓
Pretrained MobileNetV2
   ↓
Global Average Pooling
   ↓
Dropout
   ↓
Dense Layer
   ↓
Dropout
   ↓
4-Class Softmax Output
```

Important configuration:

```text
Input size       : 224 × 224 × 3
Pretrained model : MobileNetV2
Weights          : ImageNet
Dense layer      : 128 neurons
Dropout          : 0.30
Output           : 4 classes
Activation       : Softmax
Optimizer        : Adam
Initial LR       : 0.001
Loss             : Categorical Crossentropy
```

MobileNetV2 preprocessing is contained inside the trained model pipeline.

---

### 7.3 EfficientNetB0

EfficientNetB0 was also implemented using transfer learning.

It provides another pretrained architecture for comparison with MobileNetV2 and the Custom CNN.

---

## 8. Model Evaluation

Each trained model was evaluated on the independent test dataset.

The following metrics were considered:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Custom CNN | 50.41% | 40.98% | 50.41% | 44.99% |
| MobileNetV2 | **84.96%** | **85.97%** | **84.96%** | **84.84%** |
| EfficientNetB0 | 83.74% | 84.62% | 83.74% | 83.62% |

Based on the recorded evaluation results, **MobileNetV2 obtained the highest test accuracy among the three implemented models** and was therefore selected for the final application.

---

## 9. Final Model

The final deployed model is:

```text
MobileNetV2
```

Saved model:

```text
models/mobilenet_best.keras
```

Final test accuracy:

```text
84.96%
```

Model input:

```text
(None, 224, 224, 3)
```

Model output:

```text
(None, 4)
```

---

## 10. Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated into the project to improve model interpretability.

The system produces:

1. Original MRI image
2. Grad-CAM activation map
3. Explainability overlay

For the MobileNetV2 model, the feature extractor produces feature maps with shape:

```text
(None, 7, 7, 1280)
```

The Streamlit application visualizes stronger positive contributions using warmer areas of the activation map.

### Important Interpretation

Grad-CAM explains which image regions contributed more strongly to the selected classification.

It **must not** be interpreted as:

- Tumor segmentation
- Exact tumor boundary detection
- Clinical localization
- Medical diagnosis

Grad-CAM is used only as a model explainability technique in this project.

---

## 11. Streamlit Web Application

An interactive Streamlit interface was developed for demonstrating the final model.

The application allows the user to:

- Upload a brain MRI image
- Preview the uploaded MRI
- Run MobileNetV2 inference
- Display the predicted class
- Display prediction confidence
- View probabilities for all four classes
- Generate a Grad-CAM activation map
- Display the Grad-CAM overlay
- Inspect technical model information

The interface also contains dedicated sections for:

- Analysis
- Performance
- Dataset
- Methodology

---

## 12. Application Workflow

```text
MRI Image
    ↓
Image Upload
    ↓
RGB Conversion
    ↓
Resize to 224 × 224
    ↓
MobileNetV2 Model
    ↓
Softmax Probabilities
    ↓
Predicted Class + Confidence
    ↓
Grad-CAM
    ↓
Explainability Visualization
```

---

## 13. Project Structure

```text
BRAIN-TUMOR-MRI-CLASSIFICATION/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   └── mobilenet_best.keras
│
├── notebooks/
│   └── brain_tumor_mri_analysis.ipynb
│
├── results/
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── figures/
│   ├── gradcam/
│   ├── custom_cnn_history.json
│   ├── custom_cnn_test_metrics.json
│   ├── efficientnet_history.json
│   ├── efficientnet_test_metrics.json
│   ├── mobilenet_history.json
│   ├── mobilenet_test_metrics.json
│   ├── model_comparison.json
│   └── model_comparison_report.txt
│
├── src/
│   ├── compare_models.py
│   ├── custom_cnn.py
│   ├── data_loader.py
│   ├── dataset_analysis.py
│   ├── efficientnet_model.py
│   ├── evaluate_custom_cnn.py
│   ├── evaluate_efficientnet.py
│   ├── evaluate_mobilenet.py
│   ├── gradcam.py
│   ├── mobilenet_model.py
│   ├── predict.py
│   ├── train_custom_cnn.py
│   ├── train_efficientnet.py
│   ├── train_mobilenet.py
│   └── visualize_dataset.py
│
├── README.md
└── requirements.txt
```

---

## 14. Installation

### Step 1 — Clone or open the project

Open the project directory in VS Code.

### Step 2 — Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
```

### Step 3 — Activate the environment

```bash
source .venv/bin/activate
```

### Step 4 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 15. Running the Streamlit Application

From the project root directory, activate the virtual environment:

```bash
source .venv/bin/activate
```

Then start the application:

```bash
streamlit run app/app.py
```

Streamlit will start a local development server and provide the local application address in the terminal.

Upload an MRI image and select:

```text
Run MRI Analysis
```

The system will generate the prediction, class probabilities, and Grad-CAM visualization.

---

## 16. Running the Analysis Notebook

Open:

```text
notebooks/brain_tumor_mri_analysis.ipynb
```

Select the project's `.venv` Python environment as the Jupyter kernel.

If `ipykernel` is not installed:

```bash
python -m pip install ipykernel
```

Then select:

```text
.venv
```

as the notebook kernel and use:

```text
Run All
```

to execute the notebook.

---

## 17. Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit
- Jupyter Notebook
- VS Code
- MobileNetV2
- EfficientNetB0
- Convolutional Neural Networks
- Grad-CAM

---

## 18. Key Features

- Four-class MRI classification
- Custom CNN baseline
- MobileNetV2 transfer learning
- EfficientNetB0 transfer learning
- Model performance comparison
- Test-set evaluation
- Confusion matrices
- Classification reports
- Prediction confidence
- Class probability visualization
- Grad-CAM explainability
- Interactive Streamlit interface
- Reproducible Jupyter analysis
- Organized experiment results

---

## 19. Limitations

This project has several important limitations.

The model's performance depends on the characteristics and quality of the dataset used for training and evaluation.

The system has not been validated for clinical deployment.

Grad-CAM provides an explanation of model attention/contribution but does not provide tumor segmentation.

High softmax confidence does not guarantee that a prediction is clinically correct.

Performance on MRI images from hospitals, scanners, acquisition protocols, or populations not represented in the dataset may differ from the recorded test results.

---

## 20. Future Scope

Future improvements may include:

- Training with larger and more diverse MRI datasets
- Cross-dataset validation
- Fine-tuning additional pretrained architectures
- Vision Transformer-based classification
- Tumor segmentation using U-Net or related architectures
- Improved explainable-AI techniques
- Model calibration and uncertainty estimation
- Hyperparameter optimization
- External validation
- Deployment using cloud infrastructure
- More extensive robustness testing

---

## 21. Conclusion

This project demonstrates an end-to-end deep-learning workflow for four-class brain MRI image classification.

A Custom CNN, MobileNetV2, and EfficientNetB0 were implemented and evaluated. Among the recorded experiments, MobileNetV2 achieved the highest test accuracy of approximately **84.96%**.

The selected MobileNetV2 model was integrated into a Streamlit application that provides class predictions, confidence scores, class probability distributions, and Grad-CAM explainability.

The completed system demonstrates image preprocessing, deep learning, transfer learning, model evaluation, explainable AI, and interactive application deployment within a single academic machine-learning project.

---

## Disclaimer

**Research and educational use only.**

This application is an academic machine-learning prototype. It is not a medical device and must not be used to provide medical diagnosis, treatment advice, or clinical decision-making.