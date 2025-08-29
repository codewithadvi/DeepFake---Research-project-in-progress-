# Exposing Digital Illusions: A Comparative Analysis of CNN Architectures for Deepfake Detection

## Project Overview

This project focuses on developing a robust DeepFake detection system using **Convolutional Neural Networks (CNNs)** and other deep learning models. The goal is to identify AI-generated or manipulated facial images and videos, which are increasingly used in misinformation, identity theft, and malicious activities. The system aims to achieve high accuracy and low false-positive rates. It also enhances interpretability with Explainable AI (XAI) techniques to provide transparent and trustworthy predictions.

---

## 📊 Methodology

### Dataset Description
The dataset was created by combining two publicly available datasets from Kaggle: "140K Real/Fake Faces" and "Real/Fake Face Detection". This resulted in a balanced dataset consisting of **142,356 images** (71,178 real and 71,178 fake). All images were uniformly resized to **$224\times224$ pixels**.

Preprocessing steps included:
* **Contrast Limited Adaptive Histogram Equalization (CLAHE)** for contrast enhancement.
* Data augmentation (random shear, horizontal flips, and rotations) to improve model generalization and minimize bias.

The dataset was divided as follows:
* **80%** for training
* **10%** for validation
* **10%** for testing

### Model Architectures
We evaluated three distinct CNN architectures for the binary classification task (real or fake).

#### 1. VGGFace16
A pre-trained model initially trained on celebrity faces was used. The top dense layers were substituted with a custom classifier, which included a 512-neuron dense layer and a sigmoid output.

#### 2. DenseNet-121
This model was used in a transfer learning setup, using highly interconnected layers to extract the best features. The lower layers were frozen to prevent pre-trained representation from getting jumbled, and the last three dense blocks were fine-tuned on our dataset. Feature vectors of size 2048 were extracted and a **PCA-SVM** pipeline was used for classification.

#### 3. Custom CNN
Our self-designed 12-layer facial image classifier uses alternating convolutional and max-pooling blocks for step-wise downsampling.
* Each convolutional layer uses **$3\times3$ filters** with 'same' padding and ReLU activation, and the number of filters increases across layers (16, 32, 64, 128, 256, 512).
* **Batch Normalization** is applied after each convolutional layer for training stability.
* A **$2\times2$ max pooling** is used for spatial reduction.
* A **dropout of 0.25** is applied after some blocks to prevent overfitting.
* The final feature maps are flattened into a 1024-dimensional embedding using **Global Average Pooling**.
* A dense layer with a sigmoid activation function is used for the final binary classification output.

---

### Training Specifications
All models were trained with the following parameters:
* **Optimizer:** Adam (with a learning rate of $1e^{-4}$).
* **Loss Function:** Binary cross-entropy.
* **Batch Size:** 32.
* **Epochs:** 8 (with early stopping on validation loss).

---

## 📈 Results and Observations

### Performance Evaluation
The following table summarizes the quantitative results of the three models on our dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGGFace16** | 95.2% | 96.0% | 94.0% | 94.8% | 0.9601 |
| **DenseNet-121** | 96.8% | 97.2% | 96.0% | 96.6% | 0.9924 |
| **Custom CNN** | 95.0% | 95.5% | 94.5% | 95.0% | 0.9956 |

The custom CNN, with a remarkable ROC AUC of 0.9956, performed slightly better than the other models in this metric, while also being computationally efficient.

### Training and Validation Curves
* **VGGFace16:** The training and validation loss both decreased slowly and steadily, indicating great generalization with no significant overfitting. Accuracies for both reached a plateau above 98% by Epoch 2, showing robust performance on new data.
* **DenseNet-121:** The training loss declined steadily, but the validation loss showed significant fluctuations, suggesting some instability or potential overfitting.
* **Custom CNN:** Both training and validation accuracy increased steadily with a very small difference, indicating a perfect balance of learning and generalization and no apparent overfitting.

### PCA-SVM Analysis
We used a PCA-SVM pipeline to assess the discriminative ability of the feature representations learned by each CNN. The contour plots of the principal components revealed the following:
* **VGGFace16:** The plot showed two high-density regions with a fascinating overlapping area where classes were not completely separable.
* **DenseNet-121:** This plot also showed two distinct high-density peaks, suggesting the model captures at least two significant data modes (real and fake). The neat separation indicates good clustering, but the overlapping areas could lead to misclassifications.
* **Custom CNN:** The plot for our custom CNN showed a single, tight peak at the origin. This suggests that the most discriminative features might reside in higher-dimensional space, but it confirms that the custom CNN creates a coherent embedding space.

### Custom CNN-Specific Metrics
#### Confusion Matrix
The confusion matrix for the custom CNN model shows:
* **True Positives (TP):** 9183 (correctly identified fake images).
* **True Negatives (TN):** 9665 (correctly identified real images).
* **False Positives (FP):** 135 (real images incorrectly classified as fake).
* **False Negatives (FN):** 817 (fake images incorrectly classified as real).

This demonstrates the model's well-balanced capability to identify deepfakes with a relatively low number of false positives and negatives.

#### ROC Curve
The ROC curve for the custom CNN shows an AUC of **1.00**. This indicates a perfect classifier, as the curve is significantly above the random-guess baseline, enabling high true positive rates at very low false positive rates. This makes the model appropriate for security-critical applications.

---

## 🧠 Explainable AI (XAI) Implementation
To improve transparency and trust in model predictions, the following XAI techniques were implemented:

#### SHAP (SHapley Additive Explanations)
* SHAP highlights regions of the face that positively (red) or negatively (blue) contribute to model decisions.
* The visualization shows that the custom CNN heavily focuses on the **lower face and lips area**, confirming that the model bases its decision on semantically relevant facial features rather than irrelevant background elements.

#### LIME (Local Interpretable Model-agnostic Explanations)
* LIME provides interpretable local approximations around individual predictions.
* It highlights the facial regions that were most important for the prediction. The LIME explanations show that the model effectively identifies key facial regions, reducing noise and providing a contextual understanding of its decision.

#### Saliency, LRP, and Integrated Gradients
These techniques helped visualize spatial attention across facial regions and confirmed that critical features were consistently utilized during prediction.

---

## 📝 Conclusion and Future Work
Our research confirms that all three models, VGGFace16, DenseNet-121, and our custom CNN, are effective deepfake detectors with accuracies above 90%. Our custom CNN is particularly noteworthy for achieving performance on par with pre-trained models while being more computationally efficient. The integration of XAI techniques like SHAP and LIME successfully revealed the significant features the model uses to distinguish between real and fake faces, thereby increasing the model's interpretability and trustworthiness.

Future work will focus on:
* **Hyperparameter Tuning:** Implementing more sophisticated tuning and regularization to improve training stability and generalization.
* **Larger Datasets:** Utilizing larger datasets to further enhance detection robustness.
* **Multimodal Information:** Incorporating additional data, such as audio and metadata, to improve detection capabilities.
* **Deployment:** Deployment of the model through web applications (e.g., Hugging Face).

---

## ⚙️ Technologies and Tools
* **Frameworks:** TensorFlow, Keras, and PyTorch.
* **Techniques:** Transfer learning, hybrid models, and XAI visualization.
* **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.

---

## 📁 Key Components
* `custom_cnn.ipynb`: A custom-built CNN architecture for detecting DeepFakes.
* `densenet.ipynb`: Implementation of DenseNet for feature extraction and classification.
* `vggface.ipynb`: Utilizes VGGFace for face recognition and forgery detection.
* `DeepFake_hybrid_model_4_epochs.ipynb`: A hybrid model trained over 4 epochs, combining multiple architectures.
* `XAI_implementation.ipynb`: Incorporates Explainable AI (XAI) techniques to interpret model predictions.
* `XAI_implementation_finalized_4epoch_3-3.ipynb`: Finalized version with optimized XAI visualization for better interpretability.
* `RP_report_draft_DeepFake_detection.pdf`: Draft of the research paper, covering methodology, results, and analysis.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* TensorFlow, Keras, and PyTorch libraries.

### Installation
```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
