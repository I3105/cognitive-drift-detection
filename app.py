import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Cognitive Drift Detection", layout="wide")

st.title("🧠 Cognitive Drift Detection System")
st.markdown(
    """
This app detects **cognitive drift** in real-time patterns based on simulated cognitive metrics.
You can simulate data streams, visualize them, and detect drift automatically.
"""
)

# ---- Sidebar ----
st.sidebar.header("Simulation Settings")
num_points = st.sidebar.number_input("Number of data points", min_value=50, max_value=1000, value=200)
drift_threshold = st.sidebar.slider("Drift Threshold (z-score)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
noise_level = st.sidebar.slider("Noise Level", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

# ---- Simulate Real-time Data ----
@st.cache_data
def generate_cognitive_data(n_points=200, noise=0.2):
    """
    Simulate cognitive metric stream with occasional drift
    """
    np.random.seed(42)
    time = np.arange(n_points)
    # Base cognitive metric (e.g., attention score between 0 and 1)
    base_metric = np.sin(time/20) + np.random.normal(0, noise, size=n_points)
    
    # Introduce drift randomly
    drift_points = np.random.choice(range(50, n_points), size=3, replace=False)
    for dp in drift_points:
        base_metric[dp:] += np.random.uniform(0.5, 1.0)  # shift
    return pd.DataFrame({"Time": time, "CognitiveMetric": base_metric})

data = generate_cognitive_data(num_points, noise_level)

# ---- Detect Cognitive Drift ----
def detect_drift(df, threshold=2.0):
    """
    Detect drift using z-score over rolling window
    """
    window = 10
    df['RollingMean'] = df['CognitiveMetric'].rolling(window).mean()
    df['RollingStd'] = df['CognitiveMetric'].rolling(window).std()
    df['ZScore'] = (df['CognitiveMetric'] - df['RollingMean']) / df['RollingStd']
    df['Drift'] = df['ZScore'].abs() > threshold
    return df

data = detect_drift(data, drift_threshold)

# ---- Visualizations ----
st.subheader("Cognitive Metric Over Time")
fig1 = px.line(data, x='Time', y='CognitiveMetric', title="Cognitive Metric Stream")
fig1.add_scatter(x=data['Time'][data['Drift']], y=data['CognitiveMetric'][data['Drift']],
                 mode='markers', marker=dict(color='red', size=10),
                 name="Drift Detected")
st.plotly_chart(fig1, use_container_width=True)

# Show detected drift points
st.subheader("Detected Drift Points")
st.dataframe(data[data['Drift']][['Time','CognitiveMetric','ZScore']])

# ---- Statistics ----
st.subheader("Summary Statistics")
st.write(f"Total points: {len(data)}")
st.write(f"Total drift points detected: {data['Drift'].sum()}")
st.write(f"Average Cognitive Metric: {data['CognitiveMetric'].mean():.2f}")
st.write(f"Standard Deviation: {data['CognitiveMetric'].std():.2f}")
