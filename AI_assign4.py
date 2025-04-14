# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Clean the file by reading with cp1252 and saving as UTF-8
input_file = "insurance.csv"
output_file = "insurance_cleaned.csv"

# Read the file with cp1252 encoding, which can handle the 0xb2 byte
try:
    with open(input_file, "r", encoding='cp1252') as file:
        content = file.read()
except Exception as e:
    print(f"Error reading the file with cp1252: {e}")
    raise SystemExit("Cannot proceed without reading the dataset.")

# Write the content to a new file in UTF-8 encoding (without BOM)
with open(output_file, "w", encoding='utf-8') as file:
    file.write(content)

# Step 2: Load the cleaned dataset
try:
    df = pd.read_csv(output_file, encoding='utf-8')
    print("Successfully loaded the cleaned file.")
except Exception as e:
    print(f"Error loading the cleaned file: {e}")
    raise SystemExit("Cannot proceed without loading the dataset.")

# (1) Regression: age vs BMI
X_bmi = df[['age']]
y_bmi = df['bmi']
model_bmi = LinearRegression().fit(X_bmi, y_bmi)
r2_bmi = r2_score(y_bmi, model_bmi.predict(X_bmi))
print(f"R² score for age vs BMI: {r2_bmi:.3f}")
# Interpret the relationship
if r2_bmi > 0.5:
    print("There is a moderate to strong linear relationship between age and BMI.")
elif r2_bmi > 0.1:
    print("There is a weak linear relationship between age and BMI.")
else:
    print("There is no significant linear relationship between age and BMI.")

# (2) Regression: age vs Charges
X_charges = df[['age']]
y_charges = df['charges']
model_charges = LinearRegression().fit(X_charges, y_charges)
r2_charges = r2_score(y_charges, model_charges.predict(X_charges))
print(f"R² score for age vs Charges: {r2_charges:.3f}")
# Interpret the relationship
if r2_charges > 0.5:
    print("There is a moderate to strong linear relationship between age and charges.")
elif r2_charges > 0.1:
    print("There is a weak linear relationship between age and charges.")
else:
    print("There is no significant linear relationship between age and charges.")