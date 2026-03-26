"""
Data preprocessing and feature engineering pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load raw sensor data"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        df = df.copy()
        
        # Remove impossible values
        df.loc[df['temperature'] < 0, 'temperature'] = np.nan
        df.loc[df['pressure'] < 0, 'pressure'] = np.nan
        df.loc[df['vibration'] < 0, 'vibration'] = np.nan
        
        # Sort by timestamp and machine_id
        df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
        
        # Handle missing values by machine
        for machine_id in df['machine_id'].unique():
            mask = df['machine_id'] == machine_id
            
            # Interpolate missing values
            df.loc[mask, 'temperature'] = df.loc[mask, 'temperature'].interpolate(method='linear')
            df.loc[mask, 'pressure'] = df.loc[mask, 'pressure'].interpolate(method='linear')
            df.loc[mask, 'vibration'] = df.loc[mask, 'vibration'].interpolate(method='linear')
            
            # Fill remaining NaN with forward/backward fill
            df.loc[mask, 'temperature'] = df.loc[mask, 'temperature'].ffill().bfill()
            df.loc[mask, 'pressure'] = df.loc[mask, 'pressure'].ffill().bfill()
            df.loc[mask, 'vibration'] = df.loc[mask, 'vibration'].ffill().bfill()
        
        return df
    
    def create_features(self, df):
        """Create time-based and statistical features"""
        df = df.copy()
        feature_dfs = []
        
        for machine_id in df['machine_id'].unique():
            machine_df = df[df['machine_id'] == machine_id].copy()
            machine_df = machine_df.sort_values('timestamp').reset_index(drop=True)
            
            # Lag features (past values)
            for col in ['temperature', 'pressure', 'vibration']:
                for lag in [1, 2, 3, 6, 12]:
                    machine_df[f'{col}_lag_{lag}'] = machine_df[col].shift(lag)
            
            # Rolling window features (using only past data)
            for col in ['temperature', 'pressure', 'vibration']:
                for window in [4, 8, 24]:  # hours
                    # Mean
                    machine_df[f'{col}_rolling_mean_{window}h'] = (
                        machine_df[col].shift(1).rolling(window=window, min_periods=1).mean()
                    )
                    # Std
                    machine_df[f'{col}_rolling_std_{window}h'] = (
                        machine_df[col].shift(1).rolling(window=window, min_periods=1).std()
                    )
                    # Min
                    machine_df[f'{col}_rolling_min_{window}h'] = (
                        machine_df[col].shift(1).rolling(window=window, min_periods=1).min()
                    )
                    # Max
                    machine_df[f'{col}_rolling_max_{window}h'] = (
                        machine_df[col].shift(1).rolling(window=window, min_periods=1).max()
                    )
            
            # Exponential moving average
            for col in ['temperature', 'pressure', 'vibration']:
                machine_df[f'{col}_ema'] = machine_df[col].shift(1).ewm(span=12, adjust=False).mean()
            
            # Rate of change features
            for col in ['temperature', 'pressure', 'vibration']:
                machine_df[f'{col}_rate_of_change'] = machine_df[col].diff()
                machine_df[f'{col}_acceleration'] = machine_df[f'{col}_rate_of_change'].diff()
            
            feature_dfs.append(machine_df)
        
        df_features = pd.concat(feature_dfs, ignore_index=True)
        
        # Drop rows with NaN created by feature engineering (first few rows per machine)
        df_features = df_features.dropna().reset_index(drop=True)
        
        # Add time-based features
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['month'] = df_features['timestamp'].dt.month
        
        return df_features
    
    def prepare_train_test_split(self, df, test_size=0.2):
        """Time-based train-test split to prevent data leakage"""
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def fit_transform(self, df, target_col='failure_next_24h'):
        """Fit scaler and transform features"""
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'machine_id', target_col]]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        # Fit and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create scaled dataframe
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        df_scaled['timestamp'] = df['timestamp'].values
        df_scaled['machine_id'] = df['machine_id'].values
        
        if y is not None:
            df_scaled[target_col] = y
        
        return df_scaled
    
    def transform(self, df, target_col='failure_next_24h'):
        """Transform features using fitted scaler"""
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        df_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        df_scaled['timestamp'] = df['timestamp'].values
        df_scaled['machine_id'] = df['machine_id'].values
        
        if target_col in df.columns:
            df_scaled[target_col] = df[target_col].values
        
        return df_scaled

def create_correlation_matrix(df, output_path='outputs/plots/correlation_matrix.png'):
    """Create and save correlation matrix"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'failure_next_24h']
    
    # Limit to top 30 features by variance for readability
    if len(numeric_cols) > 30:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(30).index.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Top Features)', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrix saved: {output_path}")
    
    return corr_matrix