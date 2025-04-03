from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import altair as alt
from altair import Chart
import vegafusion

app = Flask(__name__)
CORS(app)
# Load the model
model = joblib.load("predictive_maintenance.pkl")

# Enable the VegaFusion data transformer
alt.data_transformers.enable("vegafusion")

# Helper function for feature engineering
def feature_engineering(df):
    df['Power'] = 2 * np.pi * df['Rotational Speed'] * df['Torque'] / 60
    df['temp_diff'] = df['Process Temperature'] - df['Air Temperature']
    df['Type_H'] = 0
    df['Type_L'] = 0
    df['Type_M'] = 0
    if df['Type'].values[0] == 'L':
        df['Type_L'] = 1
    elif df['Type'].values[0] == 'M':
        df['Type_M'] = 1
    else:
        df['Type_H'] = 1

    return df.drop(['Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque'], axis=1)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json
    df = pd.DataFrame([data])
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prediction
    prediction = model.predict(df)
    prediction_probability = model.predict_proba(df)

    # Prepare response
    result = {
        "prediction": int(prediction[0]),
        "probability_no_maintenance": float(prediction_probability[0][0]),
        "probability_maintenance_needed": float(prediction_probability[0][1])
    }

    return jsonify(result)
    
@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({"status": "alive"}), 200

@app.route('/plot', methods=['POST'])
def plot():
    # Get input data from request
    data = request.json
    
    # Extract and remove 'input_feature' from data
    input_feature = data.pop('input_feature', 'Air Temperature')
    
    # Convert remaining data to DataFrame
    df = pd.DataFrame([data])
    
    # Feature engineering
    df = feature_engineering(df)

    # Debug: Print the DataFrame columns and head

    
    # Load data for comparison
    data_df = pd.read_csv("predictive_maintenance.csv")
    data_df.columns = ['UDI', 'Product ID', 'Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque', 'Tool wear', 'Machine failure', 'Failure type']
    data_df = data_df.drop(['UDI', 'Product ID', 'Failure type'], axis=1)
    
    # Debug: Print columns of data_df
   

    # Predict using the model
    try:
        machine_failure_prediction = int(model.predict(df)[0])
    except Exception as e:
        return jsonify({"error": str(e)})

    # Filter data_df for plotting
    data_df = data_df[data_df['Machine failure'] == machine_failure_prediction]

    # Create plot
    def create_chart(input_feature):
        base = alt.Chart(data_df)
        
        if input_feature == 'Type':
            chart = base.mark_bar().encode(
                alt.X('Type:O'),
                alt.Y('count()', title="Counts"),
                color=alt.condition(
                    alt.datum.Type == data['Type'],
                    alt.value('#ff4c4c'),
                    alt.value('steelblue'))
            )
        else:
            chart = base.mark_bar().encode(
                alt.X(f'{input_feature}:Q', title=f'{input_feature}').bin(maxbins=15),
                alt.Y('count()', title="Frequency"),
                color=alt.value('steelblue')
            )
            rule = base.mark_rule(color='#ff4c4c').encode(
                x=alt.datum(data[input_feature]),
                size=alt.value(3)
            )
            chart = chart + rule
        
        return chart.to_dict(format='vega')

    chart_dict = create_chart(input_feature)
    
    return jsonify(chart_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

