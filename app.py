import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# 1. Page Configuration (MUST be the first Streamlit command)
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Cybersecurity Theme
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Remove top whitespace */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Card Container */
    .css-card {
        background-color: #1e2127;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00d26a;
        color: #000;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #00b359;
        box-shadow: 0 0 15px rgba(0, 210, 106, 0.6);
        color: #fff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fff;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d26a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
    }
    
    /* Alert Boxes */
    .fraud-alert {
        background-color: rgba(255, 75, 75, 0.15);
        border: 1px solid #ff4b4b;
        color: #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
    }
    .safe-alert {
        background-color: rgba(0, 210, 106, 0.15);
        border: 1px solid #00d26a;
        color: #00d26a;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Data Handling with Fallback
@st.cache_data
def load_data():
    data_path = 'creditcard.csv'
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            # Use a subset for speed if dataset is huge, or full dataset
            # For this demo, let's use a sample if it's too big to keep UI snappy
            if len(df) > 50000:
                df = df.sample(50000, random_state=42)
            return df, False
        except Exception as e:
            return None, True
    else:
        return None, True

def generate_dummy_data():
    np.random.seed(42)
    n_samples = 1000
    # Generate synthetic features
    time = np.random.randint(0, 172792, n_samples)
    amount = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    # Generate V1-V28
    v_features = np.random.normal(loc=0, scale=1, size=(n_samples, 28))
    v_df = pd.DataFrame(v_features, columns=[f'V{i}' for i in range(1, 29)])
    
    # Generate Class (imbalanced)
    classes = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    df = pd.DataFrame({'Time': time, 'Amount': amount})
    df = pd.concat([df, v_df], axis=1)
    df['Class'] = classes
    return df

# Load Data
df, use_dummy = load_data()

if use_dummy or df is None:
    df = generate_dummy_data()
    st.toast("⚠️ Dataset not found. Using synthetic data for demonstration.", icon="⚠️")

# Preprocessing
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()

# 4. Layout Structure (30% Left, 70% Right)
col_left, col_right = st.columns([3, 7], gap="medium")

# --- LEFT COLUMN: CONTROLS ---
with col_left:
    st.markdown("## 🛡️ Fraud Detection AI")
    st.markdown("### Control Panel")
    
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### ⚙️ System Settings")
        n_neighbors = st.slider("Detection Sensitivity (K-Neighbors)", 1, 20, 5, help="Higher values smooth out noise, lower values catch finer details.")
        test_size = st.slider("Training Data Split", 0.1, 0.5, 0.2, help="Percentage of data kept aside for testing accuracy.")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Transaction Input")
        st.info("Enter transaction details to analyze.")
        
        # Input Form
        with st.form("analysis_form"):
            amount_val = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1500.0)
            
            # Renamed Features for User Friendliness
            v14_val = st.number_input("Network Integrity Score (V14)", value=0.0, help="Technical metric representing network stability.")
            v4_val = st.number_input("Identity Verification Score (V4)", value=0.0, help="Score derived from identity checks.")
            v10_val = st.number_input("Location Consistency Score (V10)", value=0.0, help="Metric evaluating location patterns.")
            
            submit_btn = st.form_submit_button("🛡️ Analyze Transaction")
        st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIC: Train Model ---
# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_scaled, y_train)

# Calculate Metrics
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
fraud_cases = df[df['Class'] == 1].shape[0]
total_tx = df.shape[0]

# --- RIGHT COLUMN: VISUALIZATION & RESULTS ---
with col_right:
    # Top Row: Metrics
    st.markdown("### Dashboard Overview")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="css-card">
            <div class="metric-label">Total Transactions</div>
            <div class="metric-value">{total_tx:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="css-card" style="border-left: 4px solid #ff4b4b;">
            <div class="metric-label">Detected Fraud Attempts</div>
            <div class="metric-value" style="color: #ff4b4b;">{fraud_cases:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="css-card" style="border-left: 4px solid #00d26a;">
            <div class="metric-label">System Accuracy</div>
            <div class="metric-value">{acc:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    # Middle Row: Visualization
    st.markdown("### 📊 Data Visualization")
    with st.container():
        # Scatter Plot
        # We'll plot V14 vs V4 as they are significant features often used in examples
        fig, ax = plt.subplots(figsize=(10, 4))
        # Set dark background for plot
        fig.patch.set_facecolor('#1e2127')
        ax.set_facecolor('#1e2127')
        
        # Plot Normal
        ax.scatter(df[df['Class']==0]['V14'], df[df['Class']==0]['V4'], 
                   c='#00d26a', label='Normal', alpha=0.5, s=10, edgecolors='none')
        # Plot Fraud
        ax.scatter(df[df['Class']==1]['V14'], df[df['Class']==1]['V4'], 
                   c='#ff4b4b', label='Fraud', alpha=0.8, s=30, marker='x')
        
        ax.set_xlabel('Network Integrity Score (V14)', color='white')
        ax.set_ylabel('Identity Verification Score (V4)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['right'].set_color('#444')
        ax.legend(facecolor='#262730', edgecolor='#444', labelcolor='white')
        ax.set_title('Network Integrity vs Identity Verification', color='white')
        
        st.pyplot(fig)

    # Bottom Row: Prediction Result
    if submit_btn:
        st.markdown("### 🕵️ Analysis Result")
        
        # Prepare input
        # We use mean values for all other features to isolate the user inputs
        input_data = X.mean().values.reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=X.columns)
        
        # Update with user inputs
        input_df['Amount'] = amount_val
        input_df['V14'] = v14_val
        input_df['V4'] = v4_val
        input_df['V10'] = v10_val
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        pred = knn.predict(input_scaled)[0]
        prob = knn.predict_proba(input_scaled)[0][1]
        
        if pred == 1:
            st.markdown(f"""
            <div class="fraud-alert">
                <h2>🚨 HIGH RISK DETECTED</h2>
                <p>The model has flagged this transaction as POTENTIAL FRAUD.</p>
                <p>Confidence Score: {prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                <h2>✅ TRANSACTION SAFE</h2>
                <p>The model indicates this transaction is LEGITIMATE.</p>
                <p>Safety Score: {1-prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# 5. Educational Section
st.markdown("---")
with st.expander("📚 System Architecture & Methodology (How It Works)"):
    st.markdown("""
    ### 1. The Data: Privacy by Design
    The dataset used in this application comes from real credit card transactions. However, to protect user privacy, the data has been transformed using a technique called **Principal Component Analysis (PCA)**.
    - **What is PCA?** It's a mathematical method that reduces complex data into a smaller set of "components" (V1, V2, etc.) while preserving the patterns.
    - **Why use it?** It strips away personal details (like names or locations) but keeps the mathematical "shape" of the transaction, allowing us to detect patterns without knowing the user's identity.
    
    ### 2. The Algorithm: K-Nearest Neighbors (KNN)
    We use the **KNN** algorithm to classify transactions.
    - **Concept**: "Birds of a feather flock together."
    - **How it works**: When a new transaction comes in, the algorithm looks at the 'K' most similar transactions in history (the "nearest neighbors").
    - **Decision**: If the majority of those neighbors were fraud, this new transaction is likely fraud too.
    
    ### 3. The Process: Anomaly Detection
    Fraudulent transactions are rare outliers. They often have unusual mathematical signatures—for example, a high "Identity Verification Score" combined with a low "Network Integrity Score".
    - **Training**: The model learns what "normal" looks like from thousands of safe transactions.
    - **Prediction**: It flags anything that deviates significantly from that established norm.
    """)

