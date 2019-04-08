# For (somewhat) reproducible results
import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '9'
np.random.seed(9)
rn.seed(9)

import pandas as pd
import json
from os import listdir
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from general_constants import path_indices, path_full_set, cols

# ----------------------------------------------------------------------
# Setup constants and determine IDT or DCN
# ----------------------------------------------------------------------
# Determines output level
verbose = 2

model_folder = '../models/train_test/'
model_files = listdir(model_folder)
model_files = [model_folder + model_file for model_file in model_files]
# Read entire dataset
df = pd.read_csv(path_full_set, index_col=0)
# Keep only columns of interest & save names for later
names = df['Name']
df = df[cols]
# Initialize df variables
with open(path_indices) as json_file:
    indices = json.load(json_file)
train_indices = indices['train_indices']
test_indices = indices['test_indices']
df_train = df.iloc[train_indices]
df_test = df.iloc[test_indices]
x_train, y_train = df_train.iloc[:, 0:-1].values, df_train.iloc[:, -1].values
x_test, y_test = df_test.iloc[:, 0:-1].values, df_test.iloc[:, -1].values
input_dim = x_train.shape[1]

# Standardize
scaler = StandardScaler().fit(x_train)
x_test = scaler.transform(x_test)

# Get predictions of each model
dfs = []
for model_file in model_files:
    model = load_model(model_file)
    # Inference and error calculation
    predictions = model.predict(x_test).flatten()
    # Save predictions
    df_test['Prediction'] = predictions
    # Make a copy
    df = df_test.copy()
    dfs.append(df)
# Get average prediction
df_average = pd.concat(dfs).groupby(level=0).mean()
# Get error metrics
predictions = df_average['Prediction']
difference = y_test - predictions
error_a = abs(difference)
error_p = abs(np.divide(difference, y_test)) * 100
error_ma = np.average(error_a)
error_rms = np.sqrt(np.mean(np.square(difference)))
error_mp = np.mean(error_p)

# Save relevant error metrics
df_average['Absolute difference'] = error_a
df_average['Percentage difference'] = error_p

# Print average errors
print({'error_ma': error_ma, 'error_rms': error_rms, 'error_mp': error_mp})

# Replace index with name of compound
df_average['Name'] = names
df_average.set_index(keys='Name', inplace=True)

# Save the sheet
df_average.to_csv('../results/train_test_average.csv')
