# Intelligent X-Ray Pneumonia Detection

> **Empowering clinicians with fast, explainable, and accurate chest X-ray diagnostics.**

## 🌍 Overview
Pneumonia remains one of the leading causes of preventable death globally. While chest X-rays are the gold standard for diagnosis, interpreting them requires highly trained radiologists, who are often in short supply in resource-constrained environments. 

This repository contains the visual diagnostic engine for **Pneumo AI**—a deep learning framework designed to analyze chest X-rays and instantly detect the presence of pneumonia. Built with PyTorch, this tool doesn't just deliver a binary "yes" or "no"; it utilizes **Explainable AI (XAI)** to highlight *where* it sees the anomaly, allowing medical professionals to make faster, more confident decisions.

---

## ✨ Key Features
* **Robust Diagnostic Backbone:** Utilizes a fine-tuned `ResNet18` convolutional neural network to extract deep visual features from complex medical imagery.
* **Explainable AI (Grad-CAM):** Medical AI must be transparent. We integrated Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heat maps, visually explaining which regions of the lungs drove the model's prediction.
* **Automated DICOM Processing:** Includes a custom data pipeline to seamlessly convert raw, clinical-grade DICOM (`.dcm`) medical files into normalized, model-ready images.
* **Interactive Clinician Interface:** Features a lightweight, shareable web application built with Gradio, allowing users to intuitively upload X-rays and receive instant probabilistic feedback.

---

## 🛠️ Technical Architecture

### 1. The Dataset
This model is trained on the **RSNA Pneumonia Detection Challenge** dataset. The data pipeline handles:
* Parsing `.dcm` files directly via the Kaggle API.
* Normalizing pixel values and converting them to 3-channel standard images.
* Implementing stratified splitting to maintain class distribution across training and validation sets.

### 2. Model & Training
* **Architecture:** `timm`-based ResNet18 (pretrained).
* **Optimization:** `AdamW` optimizer paired with a Cosine Annealing Learning Rate Scheduler for smooth, stable convergence.
* **Hardware Efficiency:** Implements **Gradient Accumulation**, allowing the model to simulate larger batch sizes and train effectively even on memory-constrained GPUs (like the Colab T4).

---

## 🚀 Getting Started

### Prerequisites
You will need a GPU-enabled environment (like Google Colab) and an active Kaggle account to fetch the dataset. 

Ensure you have your `kaggle.json` API token ready. **Important:** You must visit the RSNA Pneumonia Detection Challenge page on Kaggle and click "I Understand and Accept" on the competition rules before the API will allow you to download the data.

### Installation
1. Clone this repository or open the provided Jupyter Notebook.
2. Install the required dependencies:
   ```bash
   pip install torch torchvision transformers datasets accelerate timm kaggle pydicom opencv-python scikit-learn matplotlib seaborn grad-cam gradio


### USAGE

### Training the Model:
Run the notebook cells sequentially. The training loop will automatically:

Download and extract the RSNA dataset.

Preprocess the images and generate a metadata CSV.

Train the ResNet18 model across the specified epochs, evaluating Accuracy, Recall, F1-Score, and AUC.

Save the best-performing weights as best_resnet18.pth.

### Launching the Dashboard:
Once the model is trained, simply execute the Gradio cell. It will generate a public link. You can share this link with others to access the web interface. Upload a .png or .jpg X-ray, and the dashboard will output the probability distribution between "Normal" and "Pneumonia".

### FUTURE ROADMAP
• Advanced Data Augmentation: Integrating more aggressive medical-specific augmentations (like elastic transforms) to improve generalization on edge cases.

• Class Weighting: Implementing dynamically calculated loss weights to better handle the natural imbalance between negative and positive cases.

• Cloud Deployment: Containerizing the Gradio app with Docker for permanent hosting on AWS or Google Cloud.

 #### Disclaimer: This software is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.


