import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load and preprocess data
df = pd.read_csv('Final_Data1.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')

# Define the conditions for servicing
df['Needs_Service'] = ((df['Threshold'].isin(['High', 'Low'])) & (df['Probability of Failure'] == 'High')).astype(int)

# Encode categorical features
label_encoders = {}
for column in ['Machine', 'Component', 'Parameter']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df[['Machine', 'Component', 'Parameter', 'Value']]
y = df['Needs_Service']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
service_model = RandomForestClassifier(n_estimators=100, random_state=42)
service_model.fit(X_train, y_train)

# Print model performance
y_pred = service_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the model and encoders
with open('service_model.pkl', 'wb') as model_file:
    pickle.dump(service_model, model_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)