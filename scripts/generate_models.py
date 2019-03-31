# For (somewhat) reproducible results
import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '9'
np.random.seed(9)
rn.seed(9)

import pandas as pd
import json
from keras.models import save_model
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from general_constants import loss_options, path_indices, path_full_set, cols


def model2(input_dim, loss, r1, l1, r2, l2):
    """
    Returns a two layer artificial neural network model with the number of units
    defined by l1 and l2, and regularization coefficients by r1 and r2.
    """
    model = Sequential()
    model.add(Dense(l1, input_dim=input_dim, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(r1))
    model.add(Dense(l2, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(r2))
    model.add(Dense(1, kernel_initializer='he_uniform'))
    model.compile(loss=loss, optimizer='adam')
    return model


# ----------------------------------------------------------------------
# Setup constants and determine IDT or DCN
# ----------------------------------------------------------------------
loss = loss_options[2]
batch_size = 32
epochs = 3

# Dropout coefficients
dropout_1 = 0.135
dropout_2 = 0.041

# Layer sizes
nodes_1 = 442
nodes_2 = 290

# Determines output level
verbose = 2

for run in ['train_test', 'full']:
    # Read entire dataset
    df = pd.read_csv(path_full_set, index_col=0)
    # Keep only columns of interest
    df = df[cols]
    # Initialize df variables
    df_train, df_test = None, None
    if run == 'train_test':
        with open(path_indices) as json_file:
            indices = json.load(json_file)
        train_indices = indices['train_indices']
        test_indices = indices['test_indices']
        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]
    elif run == 'full':
        df_train = df
        df_test = df

    x_train, y_train = df_train.iloc[:, 0:-1].values, df_train.iloc[:, -1].values
    x_test, y_test = df_test.iloc[:, 0:-1].values, df_test.iloc[:, -1].values
    input_dim = x_train.shape[1]

    # Standardize
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    for i in range(10):
        # Set up model and fit
        model = model2(input_dim, loss, dropout_1, nodes_1, dropout_2, nodes_2)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Inference and error calculation
        predictions = model.predict(x_test).flatten()
        difference = y_test - predictions
        error_ma = np.average(abs(difference))
        error_rms = np.sqrt(np.mean(np.square(difference)))
        error_mp = np.mean(abs(np.divide(difference, y_test))) * 100

        if verbose > 0:
            print("Mean absolute error:          {:.3f}".format(error_ma))
            print("Root mean squared error:      {:.3f}".format(error_rms))
            print("Mean percentage error:        {:.3f}%".format(error_mp))

        if verbose > 1:
            difference_percentage = abs(np.divide(difference, y_test)) * 100
            combined = zip(y_test, predictions, difference, difference_percentage)
            print("  y_test  preds   diff    diff_%")
            for a, b, c, d in combined:
                print("{:7.2f} {:7.2f} {:7.2f} {:7.2f}".format(a, b, c, d))

        # Divide by number of folds to get the average error
        errors = {'error_ma': error_ma,
                  'error_rms': error_rms,
                  'error_mp': error_mp}

        print(errors)
        print(model.summary())

        # Save each model, name according to percentage error
        error_string = '{:.2f}'.format(error_mp).replace('.', '_')
        save_model(model, "../models/{}/model_{}.h5".format(run, error_string))

        # Save predictions
        df_test['Prediction'] = predictions
        df_test.to_csv('../results/{}/results_test_set_{}.csv'.format(run, error_string))