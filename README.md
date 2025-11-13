# Malaria Parasite Detection using Deep Learning

This repository contains the code and trained models for an automated **malaria parasite detection system** built using **PyTorch** and **ResNet50**.  
Given a microscopic image of a thin blood smear, the model classifies the cell as:

- **Parasitized (malaria positive)**  
- **Uninfected (malaria negative)**  

The project started as a B.Tech project at **Vellore Institute of Technology (VIT), Vellore** under the guidance of **Dr. Mohanasundaram R**.

---

## Dataset

This code uses the **official NIH Malaria Cell Images Dataset**:

> **NIH Malaria Dataset:**  
> https://ceb.nlm.nih.gov/repositories/malariadatasets/

- ~27,500 cell images
- Two classes: **Parasitized** and **Uninfected**
- RGB images of individual cells cropped from thin blood smear slides

Download the dataset from the NIH website and place it in a suitable directory (e.g. `data/`), keeping separate folders for `Parasitized` and `Uninfected` as expected by the training code.

---

## Project Goals

1. Build a **fully automated image classification pipeline** for malaria detection.
2. Use a **ResNet-based deep neural network** for high classification accuracy.
3. Provide simple **inference scripts / notebooks** so that any given blood cell image can be classified as infected or not.
4. Improve on traditional image-processing-only approaches by leveraging **deep learning**.

---

## Repository Structure

```text
Image-malaria-prediction/
│
├── README.md                       # Project documentation (this file)
├── .gitattributes
├── kaggle.json                     # Example Kaggle API credentials file (do NOT commit your own)
├── main image project-converted.pdf# Project report / documentation
│
├── malaria_detection_checkpoint.ipynb      # Original training / experimentation notebook
├── malaria_detection_checkpoint-2.ipynb    # Additional training / experimentation notebook
├── malaria_detection_checkpoint-3.py       # Python script version of training pipeline
│
├── malaria_app.ipynb               # Notebook for running the classifier as an app / demo
├── malaria_app.py                  # Inference / app script (classify sample images)
├── malaria_app_main.py             # Main entry point for app-style execution
│
├── malaria_resnet50_best.pth       # Trained ResNet50 model weights
└── requirements.txt                # Python dependencies
```
# Methodology
## 1. Preprocessing & Data Loading
Load images from the NIH dataset folders.
Resize and normalize images to match ResNet50 input requirements.
Apply basic transformations (e.g., resizing, tensor conversion, normalization).
## 2. Model Architecture – ResNet50
Base architecture: Residual Neural Network (ResNet50).
Residual connections (skip connections) help train deeper networks by avoiding vanishing gradients.
Final fully-connected layer is adapted to output 2 classes (Parasitized / Uninfected).
## 3. Training
Typical training loop (see malaria_detection_checkpoint.ipynb or malaria_detection_checkpoint-3.py):
Loss function: Cross-entropy
Optimizer: (e.g.) Adam / SGD
Evaluation metrics:
Accuracy
Confusion matrix over Parasitized vs Uninfected
The best model is saved as: malaria_resnet50_best.pth

# 4. Inference
The app scripts (malaria_app.py, malaria_app_main.py) load the trained model and run prediction on new images:
Load image from path.
Apply same preprocessing as training.
Run through ResNet50 model.
Output "Parasitized" or "Uninfected", with probability scores.

# Setup & Installation
## 1. Clone the Repository
git clone https://github.com/Kartikay77/Image-malaria-prediction.git
cd Image-malaria-prediction

## 2. Create and Activate a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On macOS / Linux
# venv\Scripts\activate       # On Windows

## 3. Install Dependencies
All core dependencies are listed in requirements.txt:
pip install -r requirements.txt


### At minimum, we will need:
Python 3.x
PyTorch
NumPy
Matplotlib
scikit-image / torchvision (depending on code paths used)

# Results
From experiments reported in the project:
The system achieves high accuracy in distinguishing Parasitized vs Uninfected cells.
Residual networks (ResNet50) significantly improve performance compared to traditional image-processing-only methods.
The model is robust to variations in cell appearance and provides consistent predictions across the dataset.
(Exact accuracy values can be added here from your latest training run.)
# Future Work
## Possible extensions:
Detect different life stages of malaria parasites (e.g., trophozoite, schizont, gametocyte).
Extend to other diseases detectable from blood smear images (e.g., Dengue, H1N1–related abnormalities) using similar pipelines.
Deploy as a web or mobile app for clinical decision support.
Integrate explainability (e.g., Grad-CAM) to highlight regions that influenced the prediction.
