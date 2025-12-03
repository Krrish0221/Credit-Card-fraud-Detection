# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

### 🔴 [Live Demo: Click Here to Test the App](INSERT_YOUR_HUGGING_FACE_LINK_HERE](https://huggingface.co/spaces/Krrish0221/CreditCard)

## 📖 Project Overview
This project is a **Machine Learning-powered Web Application** designed to detect fraudulent credit card transactions. It utilizes a **K-Nearest Neighbors (KNN)** algorithm trained on a dataset of transaction records to classify activities as either "Safe" or "Fraudulent" in real-time.

The application addresses the challenge of **Imbalanced Datasets** in fraud detection, where fraudulent cases are extremely rare compared to normal transactions.

## ✨ Key Features
* **Interactive Dashboard:** Visualizes dataset balance, total transactions, and system accuracy (99.96%).
* **Real-time Prediction Simulation:** Users can manually input transaction parameters (Amount, Network Integrity, etc.) to simulate different fraud scenarios.
* **Data Visualization:** A dynamic scatter plot (using Matplotlib/Seaborn) showing the separation between safe (Green) and fraudulent (Red) transactions based on PCA features.
* **Model Tuning:** Sidebar controls to adjust model sensitivity (K-Neighbors) and training data split.

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (KNN Classifier)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 🧠 Model Insights
The model uses anonymized features from credit card transactions (often referred to as PCA vectors). For the purpose of this interface, these features have been interpreted as:
* **Network Integrity Score (V14):** Indicates the security level of the transaction network.
* **Identity Verification Score (V4):** Represents the confidence level in the user's identity.
* **Location Consistency Score (V10):** Checks if the transaction location aligns with user history.

## 🚀 How to Run Locally
If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 👨‍💻 Author
**Krish Prajapati**
*CSE - AI & ML Student*

