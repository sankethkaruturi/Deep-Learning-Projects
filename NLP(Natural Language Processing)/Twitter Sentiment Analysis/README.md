# 📢 Twitter Sentiment Analysis

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)


## 🔎 Problem Statement
Understanding public sentiment on social media platforms like Twitter is crucial for businesses, policymakers, and researchers. Sentiment analysis can provide insights into user opinions, allowing companies to improve products, services, and marketing strategies based on customer feedback. This project aims to classify tweets as positive, negative, or neutral using machine learning techniques.


## 🛠 Machine Learning Approach
This project applies machine learning and text analysis techniques to predict the sentiment of tweets. The pipeline includes:
1. **Text Preprocessing** – Tokenization, stopword removal, and vectorization.
2. **Feature Extraction** – Using NLP techniques to convert text into numerical representations.
3. **Deep Learning Models** – Training neural networks to classify sentiment.

## 🔡 Natural Language Processing (NLP)
NLP plays a crucial role in sentiment analysis by transforming textual data into structured numerical formats. This enables machine learning models to process and analyze linguistic features efficiently.

## 📏 Text Vectorization
To make textual data understandable for machine learning models, we use vectorization methods, including:
- **[Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)** – Converts text into a frequency matrix.
- **[TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)** – Weighs words based on importance within the document.

## 🤖 Machine Learning Models
This project primarily employs:
- **[Deep Neural Networks](https://www.tensorflow.org/api_docs/python/tf/keras/Model)** – A powerful technique for sentiment classification.

## 📊 Exploratory Data Analysis (EDA)
Analyzing the dataset helps us uncover patterns in sentiment distribution. Key insights include:
- A higher frequency of neutral tweets compared to positive or negative tweets.
- Common words in positive and negative tweets identified using word clouds.

### Word Clouds for Sentiment Analysis

## 🔧 Hyperparameter Tuning
Fine-tuning hyperparameters is crucial to optimizing model performance. This project explores various configurations to achieve the best results.

## 📈 Model Performance
The loss discrepancy between training and test data suggests overfitting. Despite this, the model demonstrates strong generalization on unseen data.

## 📥 How to Download and Run the Repository
### Prerequisites
- **Git** – Install from [here](https://git-scm.com/downloads).
- **Python (>=3.7)**
- **Jupyter Notebook**
- Install dependencies:
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
   cd twitter-sentiment-analysis
   ```
3. **Launch Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
4. **Run the notebook** to analyze sentiment in tweets.

---
This project provides a comprehensive framework for analyzing sentiments in tweets using machine learning and NLP techniques. 🚀

