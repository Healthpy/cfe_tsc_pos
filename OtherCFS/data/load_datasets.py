# data/load_datasets.py
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def load_ucr_dataset(dataset_name):
    """ Load a specific dataset from the UCR/UEA archives """
    ucr = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
    
    # Encode labels as integers
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # Reshape for time series (samples, features, time_steps)
    # UCR datasets are univariate, so features=1
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
    
    return X_train, y_train, X_test, y_test

def load_custom_dataset(data_path, 
                       feature_columns=None,
                       label_column=None,
                       time_column=None,
                       id_column=None,
                       test_size=0.2,
                       normalize=True,
                       resample=None):
    """
    Load and preprocess a custom time series dataset.
    
    Args:
        data_path: Path to the data file (csv or excel)
        feature_columns: List of column names containing features
        label_column: Name of the column containing labels
        time_column: Name of the column containing timestamps
        id_column: Name of the column containing series IDs
        test_size: Fraction of data to use for testing
        normalize: Whether to normalize features
        resample: Frequency for resampling (e.g., '1H', '1D')
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    # Load data based on file extension
    file_ext = os.path.splitext(data_path)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(data_path)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Identify columns if not specified
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in 
                         [label_column, time_column, id_column]]
    
    # Convert timestamp column if exists
    if time_column and time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        if resample:
            df = df.set_index(time_column).resample(resample).mean().reset_index()
    
    # Handle missing values
    df[feature_columns] = df[feature_columns].fillna(method='ffill').fillna(method='bfill')
    
    # Split into features and labels
    X = df[feature_columns].values
    y = df[label_column].values if label_column else None
    
    # Normalize features
    if normalize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    # Reshape data based on ID column if exists
    if id_column and id_column in df.columns:
        series_dict = {}
        for series_id in df[id_column].unique():
            mask = df[id_column] == series_id
            series_dict[series_id] = X[mask]
        
        # Convert to 3D array (samples, features, timesteps)
        X = np.stack([series for series in series_dict.values()])
        if y is not None:
            y = np.array([df[df[id_column] == series_id][label_column].iloc[0] 
                         for series_id in series_dict.keys()])
    else:
        # Reshape to 3D if not already
        X = X.reshape(1, X.shape[1], X.shape[0])
        if y is not None:
            y = np.array([y[0]])
    
    # Encode labels
    if y is not None:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split into train and test sets
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        y_train = y_test = None
    
    return X_train, y_train, X_test, y_test


