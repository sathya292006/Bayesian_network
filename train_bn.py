import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load dataset
df = pd.read_csv("heart.csv")

# Convert continuous to categorical (important for BN)
df['age'] = pd.cut(df['age'], bins=3, labels=[0,1,2])
df['chol'] = pd.cut(df['chol'], bins=3, labels=[0,1,2])
df['trestbps'] = pd.cut(df['trestbps'], bins=3, labels=[0,1,2])

# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('chol', 'target'),
    ('trestbps', 'target')
])

# Train model
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

# -------- Prediction --------
def predict_heart_disease(age, sex, cp, chol, trestbps):
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
    return result

# Example
print("\n🩺 Example Prediction:\n")

res = predict_heart_disease(age=1, sex=1, cp=2, chol=1, trestbps=1)
print(res)

# -------- GRAPH VISUALIZATION --------
G = nx.DiGraph()
G.add_edges_from(model.edges())

plt.figure(figsize=(6,4))
pos = nx.spring_layout(G)

nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
plt.title("Bayesian Network Structure (Heart Disease)")
plt.show()