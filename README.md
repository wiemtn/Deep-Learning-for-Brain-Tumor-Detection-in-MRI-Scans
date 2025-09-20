# Deep Learning for Brain Tumor Detection in MRI Scans

This project demonstrates a complete deep learning workflow for detecting brain tumors in MRI scans using Convolutional Neural Networks (CNNs). The notebook guides you through data exploration, preprocessing, model building, training, evaluation, and comparison of custom and pretrained models.

## Table of Contents
- [About the Data](#about-the-data)
- [Imports & Setup](#imports--setup)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Data Visualization](#data-visualization)
- [Data Processing](#data-processing)
- [First CNN Model](#first-cnn-model)
- [Pretrained CNN Model](#pretrained-cnn-model)
- [Comparison of Models](#comparison-of-models)
- [End of Notebook](#end-of-notebook)

## Project Overview
- **Goal:** Classify MRI brain images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.
- **Approach:**
  - Implement a custom CNN from scratch.
  - Apply transfer learning using a pretrained ResNet50 model.
  - Compare the performance of both models.

## Dataset
- Combined from multiple sources (e.g., figshare, Br35H, Kaggle).
- Contains 7023 MRI images labeled into four classes.
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Features
- Data exploration and visualization
- Image preprocessing and augmentation
- Custom CNN and transfer learning with ResNet50
- Model evaluation with accuracy, precision, recall, and F1-score
- Visual comparison of results

## Requirements
- Python 3.7+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV
- tqdm

Install dependencies with:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python tqdm kagglehub
```

## Usage
1. Download the dataset from Kaggle or use the provided KaggleHub integration in the notebook.
2. Open `deep.ipynb` in Jupyter or VS Code.
3. Run each cell in order to reproduce the workflow and results.

## Results
- The notebook provides a detailed comparison of a custom CNN and a pretrained ResNet50 model.
- Includes visualizations of data distribution, sample images, and model performance metrics.

## Author
**Data Rangers**

---

For questions or contributions, please open an issue or submit a pull request.