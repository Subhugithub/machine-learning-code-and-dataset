import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Load the dataset
heartDisease = pd.read_csv(r"C:\Users\Shubham\Desktop\dicision_tree\heart.csv")

print(heartDisease)

# Replace missing values ('?') with NaN
heartDisease = heartDisease.replace('?', np.nan)

# Sample instances from the dataset
print('Sample instances from the dataset are given below')
print(heartDisease.head())

# Attributes and datatypes
print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# Define the Bayesian Model
model = BayesianModel([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Learn CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
HeartDiseasetest_infer = VariableElimination(model)

# 1. Probability of HeartDisease given evidence= restecg
print('\n1. Probability of HeartDisease given evidence= restecg')
q1 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

# 2. Probability of HeartDisease given evidence= cp
print('\n2. Probability of HeartDisease given evidence= cp')
q2 = HeartDiseasetest_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)
