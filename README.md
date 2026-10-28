# Retail Product Classification Using Deep Learning

An end-to-end Computer Vision project that classifies retail products from store shelf images using a Convolutional Neural Network (CNN).

The project uses a dataset of **9,000 product images across 18 categories** to train and evaluate an image classification model.

---

## 📌 Project Overview

This project focuses on building an image classification system that identifies product categories from retail shelf images.

The workflow covers the complete AI development lifecycle, including dataset preparation, image preprocessing, CNN model development, training, evaluation, and deployment through a Streamlit web application.

The trained model predicts the product category from an uploaded image and displays the classification result.

## 🎯 Project Objectives

- Develop an end-to-end image classification pipeline.
- Prepare and structure a labeled image dataset.
- Implement CNN-based image classification.
- Train and evaluate a deep learning model.
- Analyze model performance using classification metrics.
- Deploy the trained model through a web application.

## 📊 Dataset Description

| Property | Description |
|---|---|
| Dataset Type | Retail Product Images |
| Total Images | 9,000 |
| Number of Classes | 18 |
| Data Format | Image files |
| Labels | Product category names |

Each class folder contains labeled product images used for image classification.

## ✨ Key Features

- **Multi-Class Image Classification:** Classifies retail products into 18 categories.
- **CNN-Based Model:** Uses deep learning for visual feature extraction and classification.
- **Image Preprocessing:** Resizes and normalizes images before model training and inference.
- **Model Evaluation:** Analyzes classification performance using multiple evaluation metrics.
- **Prediction Visualization:** Displays model predictions on sample images.
- **Streamlit Application:** Provides a user interface for uploading images and viewing predictions.

## 🔄 Project Workflow

```text
Raw Product Images
        ↓
Dataset Preparation
        ↓
Image Preprocessing
        ↓
CNN Model Development
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Prediction Visualization
        ↓
Streamlit Deployment
```

## ⚙️ Implementation Details

### 1. Dataset Preparation

- Downloaded the image dataset from the provided Google Drive link.
- Organized images into training, validation, and testing folders.
- Verified image labels and class folder structure.

### 2. Image Preprocessing

- Resized images to the required input dimensions.
- Normalized pixel values to the range of 0–1.
- Applied data augmentation techniques:
  - Rotation
  - Zoom
  - Horizontal flipping
  - Brightness variation

### 3. Model Development

- Developed a Convolutional Neural Network (CNN).
- Used convolutional and dense layers for feature extraction and classification.
- Applied a softmax output layer for multi-class prediction.

### 4. Model Training

- Trained the model using the prepared image dataset.
- Monitored training and validation accuracy and loss.
- Used early stopping to help reduce overfitting.

### 5. Model Evaluation

Evaluated the trained model using:

- Accuracy
- Precision
- Recall
- F1 Score

Generated:

- Confusion matrix
- Classification report
- Training and validation accuracy curves
- Training and validation loss curves

### 6. Prediction and Visualization

- Tested the model on unseen images.
- Visualized predictions using sample test images.
- Analyzed predicted product categories.

### 7. Deployment

Developed a Streamlit web application that allows users to:

- Upload a retail product image.
- Run the trained classification model.
- View the predicted product category.
- Display the prediction confidence score.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep learning and model development |
| NumPy | Numerical computations |
| OpenCV | Image processing |
| Matplotlib | Visualization |
| Scikit-learn | Model evaluation metrics |
| Google Colab / Jupyter Notebook | Model development and training |
| Streamlit | Web application deployment |

## 📁 Repository Contents

```text
retail_classification/
│
├── CNN_code.ipynb       # Model development and training notebook
├── CNN_model.keras       # Trained CNN model
├── app.py               # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Run the Application Locally

### Prerequisites

- Python
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ashish-wandhare/retail_classification.git
```

### 2. Navigate to the Project Directory

```bash
cd retail_classification
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser. Upload a product image to view the predicted category.

## 📈 Expected Learning Outcomes

Through this project, the following concepts were explored:

- Image preprocessing and dataset structuring.
- CNN architecture and image classification.
- Model training and validation.
- Evaluation using accuracy, precision, recall, and F1 score.
- Confusion matrix and classification report analysis.
- Training and validation curve visualization.
- Model deployment using Streamlit.

## 🔮 Future Improvements

- Increase the size and diversity of the image dataset.
- Experiment with transfer learning architectures.
- Improve classification performance on visually similar products.
- Add top-k prediction results.
- Enhance the web application interface.
- Integrate real-time product detection and classification.

## 🏁 Conclusion

This project demonstrates an end-to-end deep learning workflow for retail product image classification.

It covers the complete development lifecycle, from dataset preparation and CNN model training to performance evaluation and web application deployment.

The project provides practical experience in Computer Vision, deep learning, image classification, and model deployment.

---

## 🔗 Project Links

- **GitHub Repository:** [retail_classification](https://github.com/ashish-wandhare/retail_classification)
- **Streamlit Cloud:** [Live Demo](https://ashish-retailclassification.streamlit.app/)
- **AWS EC2:** [Live Demo](http://13.239.147.87:8503/)

## 📸 Application Screenshots
<img width="1917" height="737" alt="1_main_page" src="https://github.com/user-attachments/assets/d6d859dd-4f37-4776-84ea-553dcfa7ea9a" />
<img width="1917" height="937" alt="2_prediction" src="https://github.com/user-attachments/assets/f46d5cdf-0d30-4f01-a2c0-c339f4e9ef3a" />
<img width="1912" height="920" alt="3_prediction" src="https://github.com/user-attachments/assets/0660b0e1-0ebc-4986-a341-64aa85e61fa1" />


