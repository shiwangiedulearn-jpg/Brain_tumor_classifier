# BrainTumor AI

An AI-powered brain MRI image classification system built using deep learning and transfer learning. The application analyzes an uploaded brain MRI image and classifies it into one of four categories: Glioma, Meningioma, No Tumor, or Pituitary.

The project compares a custom CNN with a pretrained MobileNetV2 model and a fine-tuned MobileNetV2 model to determine the most effective approach for brain MRI classification.

---

## Overview

BrainTumor AI demonstrates a complete deep learning workflow:

- Dataset analysis
- Image preprocessing
- Data augmentation
- Custom CNN development
- Transfer learning using MobileNetV2
- Fine-tuning of a pretrained model
- Model evaluation
- Model comparison
- Streamlit deployment

The final selected model is a fine-tuned MobileNetV2 model, which achieved the best performance among the evaluated models.

---

## Classes

| Class | Description |
|---|---|
| Glioma | MRI images classified as glioma |
| Meningioma | MRI images classified as meningioma |
| No Tumor | MRI images with no detected tumor |
| Pituitary | MRI images classified as pituitary tumor |

---

## Dataset

The dataset contains **2,443 brain MRI images** divided into training, validation, and testing sets.

| Split | Images |
|---|---:|
| Training | 1,695 |
| Validation | 502 |
| Testing | 246 |
| **Total** | **2,443** |

The complete dataset is not included in this repository because of its size.

---

## Data Preprocessing

The images are:

- Converted to RGB format
- Resized to **224 × 224 pixels**
- Preprocessed according to the requirements of the selected model

### Data Augmentation

Training images are augmented using:

- Random rotation
- Random translation
- Random zoom

Vertical flipping was not used because arbitrary vertical inversion may not be appropriate for medical MRI images.

Brightness augmentation was tested during development but was removed when it did not improve validation performance.

---

## Models

### 1. Custom CNN

A convolutional neural network was developed from scratch using convolutional layers, batch normalization, ReLU activation, max pooling, global average pooling, a dense layer, dropout, and a softmax output layer.

**Test Accuracy: 80.48%**

### 2. MobileNetV2 Transfer Learning

A pretrained MobileNetV2 model with ImageNet weights was used as a feature extractor. The original classification head was replaced with a custom classification head for the four MRI classes.

The pretrained layers were initially frozen while the new classification head was trained.

**Test Accuracy: 82.52%**

### 3. Fine-Tuned MobileNetV2

The pretrained MobileNetV2 model was further fine-tuned by unfreezing the upper layers of the network.

A lower learning rate was used during fine-tuning to make controlled updates to the pretrained features. Batch normalization layers were kept frozen during fine-tuning.

**Test Accuracy: 88.21%**

This was the best-performing model and was selected as the final model for the application.

---

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 80.48% |
| MobileNetV2 | 82.52% |
| **Fine-Tuned MobileNetV2** | **88.21%** |

The fine-tuned MobileNetV2 improved test accuracy by:

- **7.73 percentage points** compared with the Custom CNN
- **5.69 percentage points** compared with the frozen MobileNetV2 model

Based on the evaluation results, the **Fine-Tuned MobileNetV2** model was selected for deployment.

---

## Final Model Evaluation

The final fine-tuned MobileNetV2 model achieved:

- **Validation Accuracy:** 89.04%
- **Test Accuracy:** 88.21%
- **Macro F1-Score:** 0.88
- **Weighted F1-Score:** 0.88

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 0.94 | 0.90 | 0.92 |
| Meningioma | 0.82 | 0.81 | 0.82 |
| No Tumor | 0.95 | 0.82 | 0.88 |
| Pituitary | 0.83 | 1.00 | 0.91 |
| **Macro Average** | **0.89** | **0.88** | **0.88** |
| **Weighted Average** | **0.89** | **0.88** | **0.88** |

The project also evaluates the final model using a confusion matrix and training/validation accuracy and loss curves.

---

## Application

The Streamlit application allows users to:

1. Upload a brain MRI image.
2. Process the image using the trained model.
3. Receive the predicted class.
4. View the model's confidence.
5. View the probability distribution across all four classes.

The application uses the final fine-tuned MobileNetV2 model for predictions.

---

## Deployment

The live application link will be added after deployment.

**Live Demo:** https://braintumorclassifier-nlp.streamlit.app/

---

## Project Structure

```text
Brain_tumor_classifier/
│
├── app.py
├── Brain_Tumor_Classification.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── brain_ai_visual.png
│
├── models/
│   └── brain_tumor_mobilenetv2_final.h5
│
└── sample_images/
    └── example_mri.jpg
```

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- MobileNetV2

### Data and Image Processing
- NumPy
- Pillow

### Web Application
- Streamlit

### Development Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## Training Workflow

```text
MRI Dataset
     |
     v
Dataset Analysis
     |
     v
Image Preprocessing
     |
     v
Data Augmentation
     |
     +----------------------+
     |                      |
     v                      v
 Custom CNN          MobileNetV2
     |                      |
     v                      v
 Evaluation          Transfer Learning
                            |
                            v
                       Fine-Tuning
                            |
                            v
                    Final Evaluation
                            |
                            v
                  Fine-Tuned MobileNetV2
                            |
                            v
                      Streamlit App
```

---

## Model File

The final trained model used by the application is:

```text
models/brain_tumor_mobilenetv2_final.h5
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project

```bash
cd Brain_tumor_classifier
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Run the Application

```bash
python -m streamlit run app.py
```

The application will open at the local Streamlit address shown in the terminal.

---

## Results

The experiments show that transfer learning improved classification performance compared with the custom CNN.

The fine-tuned MobileNetV2 achieved the highest test accuracy of **88.21%** and was selected as the final model for the application.

---

## Limitations

- The model is trained and evaluated on the available dataset and may not generalize to all MRI datasets.
- Performance may vary for images from different hospitals, scanners, imaging protocols, or populations.
- The model confidence score should not be interpreted as a clinical probability.
- The system has not been clinically validated.
- The application is intended for educational and demonstration purposes.

---

## Future Improvements

Potential future improvements include:

- Training with a larger and more diverse MRI dataset
- Further hyperparameter optimization
- Evaluation of additional pretrained architectures
- Model explainability using techniques such as Grad-CAM
- Improving confidence calibration
- External validation on independent datasets
- Further deployment and scalability improvements

---

## Disclaimer

**This application is an educational AI prototype and is not intended to replace professional medical diagnosis, medical advice, or clinical decision-making.**

Any medical decision should be made by a qualified healthcare professional.

---

## Author

Developed as a deep learning project for brain MRI image classification.
