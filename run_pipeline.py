"""
Main pipeline to run the entire predictive maintenance system
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import modules
from generate_data import generate_sensor_data
from src.preprocess import DataPreprocessor, create_correlation_matrix
from src.model_train import ModelTrainer, plot_feature_importance
from src.shap_explain import ShapExplainer
import joblib
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SYSTEM - FULL PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n[Step 1] Data Generation")
    print("-" * 40)
    
    if not os.path.exists('data/raw/sensor_data.csv'):
        print("Generating synthetic sensor data...")
        df = generate_sensor_data(n_machines=10, n_days=90, samples_per_day=24)
    else:
        print("Loading existing sensor data...")
        df = pd.read_csv('data/raw/sensor_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Step 2: Data preprocessing
    print("\n[Step 2] Data Preprocessing & Feature Engineering")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    print(f"After cleaning: {df_clean.shape}")
    
    # Create features
    df_features = preprocessor.create_features(df_clean)
    print(f"After feature engineering: {df_features.shape}")
    print(f"Number of features: {len([c for c in df_features.columns if c not in ['timestamp', 'machine_id', 'failure_next_24h']])}")
    
    # Train-test split
    train_df, test_df = preprocessor.prepare_train_test_split(df_features)
    print(f"Train set: {train_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Scale features
    train_scaled = preprocessor.fit_transform(train_df)
    test_scaled = preprocessor.transform(test_df)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df_features.to_csv('data/processed/processed_data.csv', index=False)
    print("Processed data saved.")
    
    # Create correlation matrix
    print("\nGenerating correlation matrix...")
    create_correlation_matrix(train_scaled)
    
    # Prepare data for modeling
    feature_cols = preprocessor.feature_columns
    X_train = train_scaled[feature_cols].values
    y_train = train_scaled['failure_next_24h'].values
    X_test = test_scaled[feature_cols].values
    y_test = test_scaled['failure_next_24h'].values
    
    print(f"\nClass distribution in training set:")
    print(f"Normal: {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")
    print(f"Failure: {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
    
    # Step 3: Model training
    print("\n[Step 3] Model Training & Evaluation")
    print("-" * 40)
    
    trainer = ModelTrainer()
    
    # Train models
    print("Training baseline model...")
    lr_model = trainer.train_baseline(X_train, y_train, X_test, y_test)
    
    print("\nTraining Random Forest...")
    rf_model = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\nTraining XGBoost with hyperparameter tuning...")
    xgb_model = trainer.train_xgboost(X_train, y_train, X_test, y_test, optimize=True)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Select best model (XGBoost)
    best_model = xgb_model
    print(f"\nSelected model: XGBoost")
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(best_model, feature_cols, top_n=20)
    
    # Save models
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(best_model, 'data/models/xgboost_model.pkl')
    joblib.dump(preprocessor, 'data/models/preprocessor.pkl')
    joblib.dump(feature_cols, 'data/models/feature_columns.pkl')
    print("Models saved.")
    
    # Step 4: SHAP Explainability
    print("\n[Step 4] Model Explainability (SHAP)")
    print("-" * 40)
    
    # Sample data for SHAP
    sample_size = min(500, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[sample_idx]
    
    # Create SHAP explainer
    print("Creating SHAP explainer...")
    explainer = ShapExplainer(best_model, X_train_sample, feature_cols)
    
    # Generate SHAP plots
    print("Generating SHAP visualizations...")
    explainer.create_summary_plot()
    explainer.create_bar_plot()
    
    # Create force plot for a test instance
    failure_indices = np.where(y_test == 1)[0]
    test_instance_idx = failure_indices[0] if len(failure_indices) > 0 else 0
    
    explainer.create_force_plot(X_test[test_instance_idx], test_instance_idx)
    explainer.create_waterfall_plot(X_test[test_instance_idx], test_instance_idx)
    
    # Create dependence plots
    print("Creating dependence plots...")
    explainer.create_dependence_plots(top_n=6)
    
    # Validate patterns and generate report
    print("Validating patterns and generating report...")
    explainer.generate_report()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review model comparison in: outputs/plots/model_comparison.png")
    print("2. Check SHAP explanations in: outputs/plots/")
    print("3. Read interpretation report: outputs/reports/shap_interpretation.md")
    print("4. Start the API server: python api/app.py")
    print("5. Open browser at: http://localhost:5000")
    print("\nTo start the API server, run:")
    print("  cd api")
    print("  python app.py")

if __name__ == "__main__":
    main()