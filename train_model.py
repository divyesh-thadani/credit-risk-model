import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib # Used for saving the model

print("--- Starting Model Training Script ---")

# --- 1. Load Data ---
try:
    df = pd.read_csv('loan_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'loan_data.csv' not found.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
print("Cleaning and preprocessing data...")

# Drop Loan_ID
df = df.drop('Loan_ID', axis=1)

# Handle Missing Values
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in ['LoanAmount', 'Loan_Amount_Term']:
     if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Convert Categorical Data to Numbers
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

print("Data cleaning complete.")

# --- 3. Separate Features and Target ---
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# --- 4. Train the Model ---
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y) # We train on the entire dataset for the final app
print("Model training complete.")

# --- 5. Save the Trained Model ---
# We save the model to a file named 'loan_model.pkl'
joblib.dump(model, 'loan_model.pkl')
print("Model saved successfully as 'loan_model.pkl'")

print("\n--- Script Finished ---")

