# 🌾 Wheat Disease Detection Using Deep Learning

## 🌍 Introduction
Wheat is one of the most widely cultivated crops and is an essential food source worldwide. However, various **wheat diseases** such as [Stripe Rust](https://smallgrains.wsu.edu/disease-resources/foliar-fungal-diseases/stripe-rust/) can significantly impact yield and quality, leading to economic losses for farmers.

This project leverages **deep learning and computer vision** to detect wheat diseases, providing an automated solution to assist farmers and agricultural experts.

## 🚜 Challenges
- **Manual identification** of wheat diseases is time-consuming and labor-intensive.
- **High variability** in disease symptoms across different climates and soil conditions.
- **Need for real-time monitoring** to prevent large-scale crop loss.

## 🔬 Computer Vision for Disease Detection
With advancements in **computer vision** and **deep learning**, automated models can now identify wheat diseases using image recognition techniques. This project employs **transfer learning** to classify wheat diseases from leaf images accurately.

## 🤖 Model and Approach
### 🔎 Transfer Learning
Instead of training a model from scratch, we utilize **pre-trained convolutional neural networks (CNNs)** for feature extraction:
- **VGG19** (best performing model)
- **ResNet50**
- **EfficientNet**

### 🏆 Outcomes
✅ **VGG19 achieved 95% accuracy** on test data.
✅ The model was deployed to classify diseases in real-time.
✅ Farmers and researchers can upload wheat leaf images for **instant disease diagnosis**.

## 🔮 Future Scope
🔹 Expand dataset with **high-resolution images** under different lighting conditions.
🔹 Integrate with **IoT-based smart farming** for automated monitoring.
🔹 Develop a **mobile application** for farmers to easily diagnose plant health.

## ⚡ Training with NVIDIA RTX 2080 GPU
- **GPU acceleration** significantly reduced training time.
- **TensorFlow and cuDNN optimizations** enhanced model performance.
- Enabled **real-time inference** for disease detection.

## 📥 How to Download and Run the Repository
### Prerequisites
- **Git** (Download from [here](https://git-scm.com/downloads))
- **Python (>=3.7)**
- **Jupyter Notebook**
- **Dependencies**: Install using
  ```sh
  pip install -r requirements.txt
  ```

### Steps to Clone and Execute
1. **Clone the repository**:
   ```sh
   git clone <REPOSITORY_LINK>
   ```
2. **Navigate to the project folder**:
   ```sh
   cd wheat-disease-detection
   ```
3. **Launch Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
4. **Run the notebook** to train and test the model.

---
This project demonstrates the power of **AI in agriculture**, providing an efficient and scalable solution for wheat disease detection. 🚀

