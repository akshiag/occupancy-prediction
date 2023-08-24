import pandas as pd


def preprocess_data(data):
    """
    Preprocess the sensor data by extracting and mapping time components.

    Parameters:
        data (pd.DataFrame): Sensor data.

    Returns:
        pd.DataFrame: Preprocessed sensor data.
    """
    # Extract time components
    data['time'] = pd.to_datetime(data['time'])
    data['hour'] = data['time'].dt.hour
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    data['day'] = data['time'].dt.day

    # Automatically map device names to numerical codes
    device_mapping = {code: int(code.split('_')[1]) for code in data['device'].unique()}
    data['device'] = data['device'].map(device_mapping)

    # Sort data and drop redundant columns
    data = data.sort_values(by=['time']).reset_index(drop=True)
    data = data.drop('time', axis=1)
    return data


def postprocess_data(data):
    """
    Postprocess sensor data by converting numerical codes back to devices.

    Parameters:
        data (pd.DataFrame): Sensor data.

    Returns:
        pd.DataFrame: Postprocessed sensor data.
    """
    # Convert time components back to datetime
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

    # Drop unnecessary columns
    data = data.drop(['year', 'month', 'day', 'hour', 'day_of_week'], axis=1)

    # Create a device mapping for converting numerical codes to device names
    device_mapping = {code: f'device_{code}' for code in data['device'].unique()}
    data['device'] = data['device'].map(device_mapping)

    # Convert activation_predicted values to 1 if original label was 0, and vice versa
    data['activation_predicted'] = [1 if label == 0 else 0 for label in data['activation_predicted']]
    return data
