"""
Flask REST API for model deployment
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import json
import time
import os
import sys
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import DataPreprocessor
import shap

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_columns = None
explainer = None

def load_models():
    """Load saved models and preprocessor"""
    global model, preprocessor, feature_columns, explainer
    
    try:
        print("Loading models...")
        model = joblib.load('data/models/xgboost_model.pkl')
        preprocessor = joblib.load('data/models/preprocessor.pkl')
        feature_columns = joblib.load('data/models/feature_columns.pkl')
        
        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = ['temperature', 'pressure', 'vibration', 'machine_id']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create a DataFrame with current reading
        current_reading = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'machine_id': data['machine_id'],
            'temperature': float(data['temperature']),
            'pressure': float(data['pressure']),
            'vibration': float(data['vibration']),
            'failure_next_24h': 0  # placeholder
        }])
        
        # Get historical data if provided (for feature engineering)
        if 'historical_data' in data and data['historical_data']:
            historical_df = pd.DataFrame(data['historical_data'])
            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
            
            # Combine historical and current
            full_df = pd.concat([historical_df, current_reading], ignore_index=True)
        else:
            # Use dummy historical data for feature generation
            full_df = current_reading.copy()
            
            # Add dummy historical rows for feature engineering
            for i in range(24, 0, -1):
                dummy_row = current_reading.copy()
                dummy_row['timestamp'] = current_reading['timestamp'].iloc[0] - pd.Timedelta(hours=i)
                # Add small noise to make features meaningful
                dummy_row['temperature'] = current_reading['temperature'].iloc[0] + np.random.normal(0, 1)
                dummy_row['pressure'] = current_reading['pressure'].iloc[0] + np.random.normal(0, 2)
                dummy_row['vibration'] = current_reading['vibration'].iloc[0] + np.random.normal(0, 0.01)
                full_df = pd.concat([dummy_row, full_df], ignore_index=True)
        
        # Preprocess and create features
        processor_temp = DataPreprocessor()
        full_df = processor_temp.clean_data(full_df)
        full_df = processor_temp.create_features(full_df)
        
        # Get the latest row (with all features)
        if len(full_df) == 0:
            return jsonify({
                'error': 'No valid data after preprocessing'
            }), 400
        
        latest_features = full_df.iloc[-1:].copy()
        
        # Select only the features used in training
        missing_features = [f for f in feature_columns if f not in latest_features.columns]
        if missing_features:
            # Add missing features with mean values
            for feat in missing_features:
                latest_features[feat] = 0.0
        
        X_predict = latest_features[feature_columns].values
        
        # Scale features
        X_scaled = preprocessor.scaler.transform(X_predict)
        
        # Make prediction
        failure_prob = model.predict_proba(X_scaled)[0]
        prediction = int(model.predict(X_scaled)[0])
        
        # Calculate SHAP values for explanation
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get top contributing features
        feature_contributions = []
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[::-1][:5]
        
        for idx in top_indices:
            feature_contributions.append({
                'feature': feature_columns[idx],
                'value': float(X_predict[0][idx]),
                'impact': float(shap_values[0][idx]),
                'importance': float(shap_abs[idx])
            })
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # in milliseconds
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'failure_predicted': bool(prediction),
                'failure_probability': float(failure_prob[1]),
                'normal_probability': float(failure_prob[0]),
                'risk_level': 'HIGH' if failure_prob[1] > 0.7 else 'MEDIUM' if failure_prob[1] > 0.3 else 'LOW'
            },
            'explanation': {
                'top_factors': feature_contributions,
                'recommendation': generate_recommendation(failure_prob[1], feature_contributions)
            },
            'metadata': {
                'machine_id': data['machine_id'],
                'timestamp': pd.Timestamp.now().isoformat(),
                'inference_time_ms': round(inference_time, 2)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'trace': error_trace
        }), 500

def generate_recommendation(failure_prob, top_factors):
    """Generate maintenance recommendations based on prediction"""
    recommendations = []
    
    if failure_prob > 0.7:
        recommendations.append("URGENT: Schedule immediate maintenance")
        recommendations.append("Stop machine operation if possible")
    elif failure_prob > 0.3:
        recommendations.append("Schedule maintenance within 24 hours")
        recommendations.append("Increase monitoring frequency")
    else:
        recommendations.append("Continue normal operation")
        recommendations.append("Maintain regular monitoring schedule")
    
    # Add specific recommendations based on top factors
    for factor in top_factors[:3]:
        feature_name = factor['feature']
        if 'temperature' in feature_name and factor['impact'] > 0:
            recommendations.append("Check cooling system and heat dissipation")
        elif 'vibration' in feature_name and factor['impact'] > 0:
            recommendations.append("Inspect bearings and mechanical components")
        elif 'pressure' in feature_name and factor['impact'] > 0:
            recommendations.append("Check pressure valves and seals")
    
    return recommendations

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV file
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'failure_next_24h' not in df.columns:
            df['failure_next_24h'] = 0
        
        # Process and predict for each machine
        results = []
        for machine_id in df['machine_id'].unique():
            machine_df = df[df['machine_id'] == machine_id].copy()
            machine_df = machine_df.reset_index(drop=True)
            
            # Preprocess
            processor_temp = DataPreprocessor()
            machine_df = processor_temp.clean_data(machine_df)
            machine_df = processor_temp.create_features(machine_df)
            
            if len(machine_df) == 0:
                continue
            
            # Add missing features
            for feat in feature_columns:
                if feat not in machine_df.columns:
                    machine_df[feat] = 0.0
            
            # Predict
            X = machine_df[feature_columns].values
            X_scaled = preprocessor.scaler.transform(X)
            predictions = model.predict_proba(X_scaled)
            
            # Store results
            for i in range(len(machine_df)):
                results.append({
                    'machine_id': machine_id,
                    'timestamp': machine_df.iloc[i]['timestamp'].isoformat(),
                    'failure_probability': float(predictions[i][1]),
                    'risk_level': 'HIGH' if predictions[i][1] > 0.7 else 'MEDIUM' if predictions[i][1] > 0.3 else 'LOW'
                })
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'total_records': len(results)
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in batch prediction: {error_trace}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("\n" + "="*60)
        print("Server starting at http://localhost:5000")
        print("="*60 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please run the pipeline first:")
        print("  python run_pipeline.py")