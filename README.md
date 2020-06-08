# dcn_prediction
This is the official repository for the "Predicting ignition quality of oxygenated fuels using nuclear magnetic resonance spectroscopy and artificial neural networks" paper. The Derived Cetane Number (DCN) is predicted using Artificial Neural Networks (ANNs), parameterized by functional groups, branching index, and molecular weight.

The hyperparameters of the final ANN were obtained using a multi-level grid search searching over:
* Number of hidden layers
* Loss function
* Batch size
* Epoch number

These hyperparameters were fed into a genetic algorithm, which was used to determine the number of nodes and dropout coefficient for each hidden layer of the ANN. 

Please refer to the paper for a much more detailed description of the methodology.

## Setup 
Anaconda was used to create the virtual environment for this project. Feel free to use one of the following commands to set up the required environment: 

From `.yml` file (this preserves all package versions and is thus recommended):  
`conda env create -f dcn_prediction.yml`

Conda commands:   
`conda create -n dcn_prediction python=3.6`   
`conda activate dcn_prediction`   
`conda install pandas keras scikit-learn`  

## Overview of files

#### `data/`
* `dataset_complete.csv`: complete dataset used for the paper
* `indices.json`: indices used for our train and test split 

#### `models/` 
* `train_test/`: ten models trained on the train set.
* `full/`: ten models trained on the entire dataset.

#### `results/`
* `train_test/`: ten `.csv` files containing the results for each train set model
* `full/`: ten `.csv` files containing the results for each model trained on the entire dataset
* `train_test_average.csv`: the results used for the paper. The predictions are an average over the predictions of each of the 10 models in `models/train_test/`. 

#### `scripts/`  
* `general_constants.py`: contains various constants that are used in the following two scripts.  
* `generate_models.py`: is used to create the 20 models in `models/`. It also outputs this information to the console.
* `infer_test_set.py`: is used to generate `train_test_average.csv`

## Authorship
All code was written by Vincent C.O. van Oudenhoven. Abdul Gani Abdul Jameel was responsible for the used data. Please refer to the publication itself for the full data source. 

## Acknoledgement 

## License

## BibTex
@article{article,
  author  = {Abdul Gani Abdul Jameel, Vincent C.O. van Oudenhoven, Nimal Naser, Abdul-Hamid Emwas, Xin Gao, S. Mani Sarathy}, 
  title   = {Predicting ignition quality of oxygenated fuels using nuclear magnetic resonance spectroscopy and artificial neural networks},
  year    = 2020
}

