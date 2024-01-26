# Heart Rate Prediction System

This repository contains the necessary files to preprocess data, train an LSTM model for predicting heart rates, and create an interface for predicting heart rates using the trained model. The model weights and scaler parameters are saved for future use.

## Files Overview

### 1. `traindata.ipynb`

This Jupyter Notebook file is dedicated to the preprocessing of the data, feature extraction, scaling, and training of the LSTM model for predicting heart rates. The process involves loading the dataset, extracting relevant features, scaling the data, defining and training the LSTM model, and finally, saving the model weights and scaler parameters.

#### Saved Files:
- `model.h5`: Contains the trained LSTM model weights.
- `scaler_params.pkl`: Pickled file containing the parameters used for scaling the input data.

### 2. `run.py`

This Python script serves as an interface for predicting heart rates using the trained model and scaler parameters saved in the previous step. When executed, it prompts the user to input a file (dataset) for heart rate prediction. The script then loads the model and scaler parameters, processes the input file, and displays the predicted heart rates.

To execute the script, use the following command:
python run.py

### 3. `file_predict.ipynb`

This Jupyter Notebook file is used to test the functionality of `run.py` by providing a small dataset (10 rows). It loads the data from the file, runs it through the prediction interface (`run.py`), and checks the results.

## Usage

1. Open and run `traindata.ipynb` to preprocess data, train the LSTM model, and save model weights and scaler parameters.
2. Execute `run.py` to interactively input a dataset file and get heart rate predictions using the trained model.
3. Use `file_predict.ipynb` to check the functionality of `run.py` with a small test dataset.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: TensorFlow, NumPy, Pandas (for `traindata.ipynb` and `file_predict.ipynb`), Pickle (for saving scaler parameters)

Make sure to install the required libraries using:
pip install tensorflow numpy pandas

