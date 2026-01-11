## Project Title :End-to-End AI Development: Product Category Classification from Raw Store Shelf Images

Taken 9000 images,18 class,,each class folder contains labeled product images used for image classification tasks.

---

##  Expected Learning Outcomes
Through this project, we learned:

- Image preprocessing and dataset structuring for classification
- CNN architecture and transfer learning concepts
- Model training, tuning, validation techniques
- Performance evaluation using:
  - Accuracy
  - Precision, Recall
  - F1 Score
- Visualizing:
  - Confusion matrix
  - Classification reports
  - Training curves
---

##  Workflow 

### 1) Dataset Preparation
- Dataset download from provided Drive link  
- Split images into train/validation/test folders  
- Verified each image belongs to correct class

### 2) Image Preprocessing
- Resize images (example: 224×224 / 256×256)
- Normalize pixel values (0–1)
- Data augmentation:
  - rotation
  - zoom
  - horizontal flip
  - brightness variation

### 3) Model Building
- CNN model from scratch 
- Dense layers + softmax output

### 4) Training & Evaluation
- Multi-epoch training with early stopping
- Monitoring loss, accuracy, validation metrics
- Generated:
  - Confusion matrix
  - Classification report

### 5) Testing & Visualization
- Tested model on unseen images
- Visualized predictions with sample test images
- Plotted training accuracy/loss curves

### 6) Deployment (Optional)
- Built Streamlit app:
  - Upload image
  - Predict product class
  - Display confidence score

---

##  Tools and Technologies

- **Programming:** Python  
- **Libraries:** TensorFlow/Keras, NumPy, OpenCV, Matplotlib, Scikit-learn  
- **IDE:** Google Colab / Jupyter Notebook  
- **Deployment:** Streamlit

---

##  Repository Contents
- `CNN_code.ipynb` → Training notebook
- `CNN_model.keras` → Trained CNN model
- `app.py` → Streamlit web app
- `requirements.txt` → Dependencies for deployment

---

##  Run the Streamlit App Locally

### Step 1: Install Requirements
```bash
pip install -r requirements.txt

