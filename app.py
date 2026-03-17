import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ---------------- LOAD DATA ----------------
df = pd.read_csv("heart.csv")

# Convert continuous → categorical
df['age'] = pd.cut(df['age'], bins=3, labels=[0,1,2])
df['chol'] = pd.cut(df['chol'], bins=3, labels=[0,1,2])
df['trestbps'] = pd.cut(df['trestbps'], bins=3, labels=[0,1,2])

# ---------------- MODEL ----------------
model = DiscreteBayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('chol', 'target'),
    ('trestbps', 'target')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)
inference = VariableElimination(model)

# ---------------- UI ----------------
st.title("🩺 Heart Disease Prediction (Bayesian Network)")
st.write("Enter patient details:")

# Inputs
age = st.selectbox("Age Group", [0,1,2])
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
chol = st.selectbox("Cholesterol Level (0-2)", [0,1,2])
trestbps = st.selectbox("Blood Pressure Level (0-2)", [0,1,2])

# ---------------- PREDICTION ----------------
if st.button("Predict"):
    result = inference.query(
        variables=['target'],
        evidence={
            'age': age,
            'sex': sex,
            'cp': cp,
            'chol': chol,
            'trestbps': trestbps
        }
    )

    st.subheader("Prediction Result")

    # Get probabilities
    prob_0 = result.values[0]
    prob_1 = result.values[1]

    # 🔴 IMPORTANT: Adjust based on dataset
    # Most datasets: 0 = No Disease, 1 = Disease
    prob_no_disease = prob_1
    prob_disease = prob_0

    # Display
    st.write(f"🟢 No Disease Probability: {prob_no_disease:.2f}")
    st.write(f"🔴 Disease Probability: {prob_disease:.2f}")

    # ---------------- DECISION LOGIC ----------------
if prob_disease >= prob_no_disease:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk")

# ---------------- GRAPH ----------------
st.subheader("📊 Bayesian Network Graph")

G = nx.DiGraph()
G.add_edges_from(model.edges())

fig, ax = plt.subplots()
pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", ax=ax)

st.pyplot(fig)