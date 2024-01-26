import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pickle

# Load the test data
test_data = pd.read_csv("sample_test_data.csv")##YOU CAN PROVIDE UR TEST DATASET HERE

# Drop unnecessary columns (similar to what you did in the training file)
column_to_drop = 'datasetId'
test_data.drop(column_to_drop, axis=1, inplace=True)

# Load the trained model
loaded_model = load_model('model.h5')

# Load the scaler parameters
scaler_params_path = 'scaler_params.pkl'
with open(scaler_params_path, 'rb') as scaler_file:
    scaler_params = pickle.load(scaler_file)

# Set up the scaler
standard_scaler = MinMaxScaler()
standard_scaler.min_, standard_scaler.scale_ = scaler_params['min_'], scaler_params['scale_']

# Preprocess the test input data
test_data_numeric = test_data.select_dtypes(exclude=['object'])
columns_to_scale = ['MEAN_RR', 'MEDIAN_RR', 'LF_NU', 'HF_NU', 'HF_LF', 'SDRR_RMSSD_REL_RR', 'HF_PCT', 'HF', 'SDSD_REL_RR', 'RMSSD_REL_RR', 'higuci', 'LF_HF', 'VLF', 'TP', 'sampen', 'SKEW', 'SKEW_REL_RR']
test_data_numeric[columns_to_scale] = standard_scaler.transform(test_data_numeric[columns_to_scale])

# Specify the features for prediction
X_test = test_data_numeric[columns_to_scale]

# Reshape the input data to match the model's input shape
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test data
predictions_test = loaded_model.predict(X_test_reshaped)

# Add 'uuid' and corresponding predictions to a new DataFrame
predictions_df = pd.DataFrame({'uuid': test_data['uuid'], 'predictions': predictions_test.flatten()})

# Print the first few rows of the predictions DataFrame
print("Predictions on Test Data:")
print(predictions_df.head(10))

predictions_df.to_csv('results.csv', index=False)