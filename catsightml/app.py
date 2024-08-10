from flask import Flask, request, render_template
import pandas as pd
import pickle
import datetime

app = Flask(__name__)

# Load the model and encoders
with open('service_model.pkl', 'rb') as model_file:
    service_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Load and preprocess data
df = pd.read_csv('Final_Data.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
df['Needs_Service'] = df['Probability of Failure'].map({'High': 1, 'Medium': 0, 'Low': 0})

def predict_service_need_and_days_left(machine, component, parameter, value, current_time):
    # Encode the inputs
    machine_encoded = label_encoders['Machine'].transform([machine])[0]
    component_encoded = label_encoders['Component'].transform([component])[0]
    parameter_encoded = label_encoders['Parameter'].transform([parameter])[0]

    input_data = [[machine_encoded, component_encoded, parameter_encoded, value]]
    
    service_prediction = service_model.predict(input_data)
    
    if service_prediction[0] == 1:
        return "Service Needed", 0  
    
    else:
        similar_cases = df[(df['Machine'] == machine_encoded) & 
                           (df['Component'] == component_encoded) & 
                           (df['Parameter'] == parameter_encoded) & 
                           (df['Value'] >= value)]
        
        if not similar_cases.empty:
            similar_cases['Days_Until_Service'] = (similar_cases['Time'] - current_time).dt.days
            average_days_left = similar_cases['Days_Until_Service'].mean()
            return "No Service Needed", max(0, round(average_days_left))
        else:
            return "No Service Needed", "Insufficient data to estimate days left"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    machine = request.form['machine']
    component = request.form['component']
    parameter = request.form['parameter']
    value = float(request.form['value'])
    
    # Use the current system time
    current_time = pd.Timestamp.now()
    
    service_status, days_left = predict_service_need_and_days_left(machine, component, parameter, value, current_time)
    
    result = {
        'service_status': service_status,
        'days_left': days_left
    }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)