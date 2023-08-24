#!/usr/bin/env python3

"""
This is a solution for the data science challenge.

Usage:
    solution.py <current time> <input file name> <output file name>

Where:
    <current time>: The current hour in HH:MM format.
    <input file name>: Path to the input CSV file containing historical sensor readings.
    <output file name>: Path to save the predictions for activation in the next 24 hours.
"""

import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from datetime import timedelta
from sklearn.metrics import f1_score
import joblib
from data_processing import preprocess_data, postprocess_data


def predict_future_activation(current_time, previous_readings):
    """
    Predict future hourly activation given previous sensor readings.

    Parameters:
        current_time (str): The current time in HH:MM format.
        previous_readings (pd.DataFrame): Historical sensor readings.

    Returns:
        pd.DataFrame: Predictions for activation in the next 24 hours.
    """
    # Generate full data with missing combinations
    full_data = clean_generate_full_data(previous_readings)
    print(len(full_data))
    # Preprocess data
    full_data = preprocess_data(full_data)

    # Split data into training and testing sets
    train_data, test_data = split_data(full_data)
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)

    # Build and train the model pipeline
    model = build_train_modelpipeline(X_train, y_train)
    evaluate_save_model(X_test, y_test, model)
    # Generate next 24 hours' timestamps and device combinations
    next_24_hours = pd.date_range(current_time, periods=24, freq="H").ceil("H")
    device_names = sorted(previous_readings.device.unique())
    xproduct = list(itertools.product(next_24_hours, device_names))
    predictions = pd.DataFrame(xproduct, columns=["time", "device"])

    # Generate predictions for activation in the next 24 hours
    predictions = generate_predictions(predictions, model, device_names)

    return predictions


def clean_generate_full_data(previous_readings):
    """
    Clean and whole data with missing combinations for the time when devices
    are not activated for each hour between start and end date.

    Parameters:
        previous_readings (pd.DataFrame): Previous sensor readings.

    Returns:
        pd.DataFrame: Whole sensor data with missing values.
    """
    # Convert time to datetime
    previous_readings['time'] = pd.to_datetime(previous_readings['time'])

    # Round down the minutes to the nearest hour
    previous_readings['time'] = previous_readings['time'].dt.floor('H')
    #Remove the duplicates
    previous_readings = previous_readings.drop_duplicates()

    # Calculate start and end dates
    start_date = previous_readings['time'].min().date()
    end_date = previous_readings['time'].max().date() + timedelta(days=1)

    # Generate hourly timestamps
    hourly_timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    hourly_timestamps = [ts for ts in hourly_timestamps if ts < pd.Timestamp(end_date)]

    # Generate all combinations of devices and timestamps
    all_combinations = [(device, time) for time in hourly_timestamps for device in previous_readings['device'].unique()]

    # Find existing combinations
    existing_combinations = set(zip(previous_readings['device'], pd.to_datetime(previous_readings['time'])))
    new_combinations = [comb for comb in all_combinations if comb not in existing_combinations]

    # Generate new rows for missing combinations
    new_rows = [{'device': device, 'time': time, 'device_activated': 0} for device, time in new_combinations]

    # Concatenate previous readings with new rows
    full_data = pd.concat([previous_readings, pd.DataFrame(new_rows)], ignore_index=True)
    return full_data


def split_data(data):
    """
    Split data into training and testing sets.

    Parameters:
        data (pd.DataFrame): Extended sensor data.

    Returns:
        pd.DataFrame: Training data.
        pd.DataFrame: Testing data.
    """
    split_month = 8
    split_day = 18
    train_data = data[(data['month'] == split_month) & (data['day'] < split_day)]
    test_data = data[(data['month'] == split_month) & (data['day'] >= split_day)]
    return train_data, test_data


def prepare_data(train_data, test_data):
    """
    Prepare training and testing data.

    Parameters:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.

    Returns:
        pd.DataFrame: X_train.
        np.ndarray: y_train.
        pd.DataFrame: X_test.
        np.ndarray: y_test.
    """
    # Prepare training and testing data
    X_train = train_data.drop('device_activated', axis=1)
    y_train = train_data['device_activated']
    X_test = test_data.drop('device_activated', axis=1)
    y_test = test_data['device_activated']

    # Convert labels from 0 to 1 and 1 to 0, it is being done so that ventilator
    # being turned off can be taken as positive class, because if room is predicted as being occupied,
    # even though it is not, then it can cost money
    positive_class = 0
    y_train = np.array([1 if label == positive_class else 0 for label in y_train])
    y_test = np.array([1 if label == positive_class else 0 for label in y_test])

    return X_train, y_train, X_test, y_test


def build_train_modelpipeline(X_train, y_train):
    """
    Build and train the machine learning pipeline.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline.
    """
    # Build a pipeline with normalization, decision tree, and Gaussian Naive Bayes
    pipeline = make_pipeline(
        Normalizer(norm="l1"),
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=15,
                                                           min_samples_split=16)),
        GaussianNB()
    )

    # Train the pipeline
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_save_model(X_test, y_test, model):
    """
    Evaluate the model using test data and print the F1 score.

    Parameters:
        X_test (pd.DataFrame): Test features.
        y_test (np.ndarray): Test labels.
        model (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
    """
    # Predict using the model
    results = model.predict(X_test)

    # Calculate and print F1 score
    f1 = f1_score(y_test, results)
    print('F1 score:', f1)

    # Optionally, we can add a condition to save the model only if it performs well
    # For example, we could use a threshold F1 score to determine whether to save the model to production
    # Create a dictionary to store the model and preprocessing and postprocessing functions
    # so that it is synced across all the application
    model_data = {
        'model': model,
        'preprocess_fn': preprocess_data,
        'postprocess_fn': postprocess_data
    }

    # Define the filename for the saved model
    model_filename = "trained_model.pkl"

    # Save the model dictionary to a .pkl file
    joblib.dump(model_data, model_filename)


def generate_predictions(predictions, model, device_names):
    """
    Generate activation predictions for the next 24 hours.

    Parameters:
        predictions (pd.DataFrame): DataFrame with columns "time" and "device" representing future time and devices.
        model (sklearn.pipeline.Pipeline): Trained machine learning pipeline.
        device_names (list): List of device names.

    Returns:
        pd.DataFrame: Activation predictions for the next 24 hours.
    """
    # Preprocess predictions data
    predictions = preprocess_data(predictions)

    # Predict activation using the trained model
    predictions["activation_predicted"] = model.predict(predictions)

    # Postprocess predictions data
    predictions = postprocess_data(predictions)

    # Set time as index
    predictions.set_index("time", inplace=True)
    return predictions



if __name__ == "__main__":
    # Parse command line arguments
    current_time, in_file, out_file = sys.argv[1:]

    # Read previous sensor readings from input CSV
    previous_readings = pd.read_csv(in_file)

    # Predict future activation and save the results to the output file
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)
