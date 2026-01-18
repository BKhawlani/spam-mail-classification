# Spam Mail Classification

This project focuses on building a Machine Learning model to classify emails as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques.

The goal is to automatically detect unwanted or malicious emails and improve email filtering systems.

---

## Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- NLTK
- TF-IDF Vectorizer
- Pyqt5

---

## Dataset
The dataset contains labeled email messages classified as spam or ham.
Each email undergoes text preprocessing before being used for model training.

---

## Data Preprocessing
- Lowercasing text
- Removing punctuation
- Removing stopwords
- Tokenization
- TF-IDF feature extraction

---

## Machine Learning Models
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)
##Ensemble Models
- Bagging Classifier (Bootstrap Aggregating)
- AdaBoost (Adaptive Boosting) 
---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---
## Results

The model achieves high accuracy in distinguishing spam from legitimate emails, demonstrating the effectiveness of NLP-based classification techniques

---
## Future Improvements

Deploy as a web application
Real-time email classification API
--- 

## Author

Bashar Alkhawlani
Computer Engineering Student | AI & Machine Learning Enthusiast

---
## How to Run
```bash
git clone https://github.com/BKhawlani/spam-mail-classification.git
cd spam-mail-classification
pip install -r requirements.txt
python Front_end.py
