# A Causal Framework to Measure and Mitigate Non-binary Treatment Discrimination

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<img src="beyond_bin_logo.png" alt="logo" width="225"/>

This repository contains the implementation for the paper "A Causal Framework To Measure And Mitigate
Non-binary Treatment Discrimination." 
## Overview
1. [Setup](#setup)
2. [Datasets](#datasets)
3. [Hyperparameter tuning causal generative model](#hparam_cgm)
4. [Running causal generative model](#run_cgm)
5. [Generating counterfactual values](#run_cf)

### Note
Training the causal normalizing flows models requires yaml parameter files.
Required parameter files can be found in the `_params/` folder. Please follow the format of these files to expand to other datasets.
Any new causal generative model can also be incorporated if it follows the structuring of the causal flows framework and uses related param files.

<a name="setup"></a>
## Setup 
Create the conda environment.
```bash
conda create --name carma python=3.9.12 --no-default-packages
conda activate carma
```
Install torch related things.
```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch_geometric==2.3.1
pip install torch-scatter==2.1.1

pip install -r requirements.txt
```

Finally, for plotting, install `tueplots`.
```bash
pip install git+https://github.com/pnkraemer/tueplots.git
```


<a name="datasets"></a>
## Datasets
Our implementation requires specific formatting of the datasets to work with our causal pipeline.
All processed data files need to be in this location: `Data/{data_name}/data.csv`.
Please refer to the [data_preprocessing_and_description](https://github.com/ayanmaj92/beyond-bin-decisions/blob/public/data_preprocessing_and_description.md) documentation for all details related to processing of the public datasets that we considered in our evaluations.

Finally, it is important to put the relevant details of the datasets in the appropriate `yaml` file. It is advised to keep one separate yaml file for each dataset under consideration.
Look at `_params/_causal_models/treatment_german_causal_nf.yaml` as an example:
```yaml
dataset:
  root: Data/
  name: treatment_german
  sem_name: dummy
  splits: [ 0.8,0.1,0.1 ]
  k_fold: 1
  shuffle_train: True
  loss: default
  scale: default
  add_noise: True
  num_sensitive: 2 
  num_covariate: 15 
  num_treatment: 3 
  categorical_dims: [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 20]
```

**Notes**
* For **any new dataset** ensure we have such a yaml parameter file.
* For each dataset, we need to know the number of sensitive features, number of covariates, and the number of treatments.
* For each dataset, we need to indicate which columns are the categorical columns since they require special handling in the implementation.
* The data file should be `data.csv` (1st row should be header with column names, columns should be properly ordered), and this CSV file should be in `Data/treatment_<data_name>/` folder. 

<a name="hparam_cgm"></a>
## Hyperparameter tuning causal generative model
Hyperparameter tuning causal generative models like the causal normalizing flows follows these steps. Note that this is based on the [original causal flows code](https://github.com/psanch21/causal-flows).
### a. Create grid params for grid search
Under `_params/_grid/causal_nf/` we have a basic structure.
The `base.yaml` file provides the hyperparameter values to loop over. These are the base things that will be put into the grid.
Specifically, we consider the following in the grid search:
```yaml
model:
  dim_inner: [ [ 16, 16, 16 , 16 ],  [ 32, 32, 32 ],  [ 16, 16, 16 ],   [ 32, 32 ],  [ 32 ] , [ 64 ] ]
optim:
  base_lr: [1e-2, 1e-3]
```
Then, for each treatment dataset, provide a new file like we have in `base_treatgerman.yaml`. This file needs to indicate the data details:
```yaml
dataset:
  name:  [ treatment_german ] # treatment_<name> as the folder name where CSV is.
  sem_name: [ dummy ] # always this.
  num_sensitive: 2 
  num_covariate: 15 
  num_treatment: 3 
  categorical_dims: [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 15, 16, 20]
```

### b. Running grid search
To run grid search on the specified grid space, use `causal_flows/generate_jobs.py` script. Give help to see options:
```bash
python causal_flows/generate_jobs.py --help
```
For instance, one can run it as:
```bash
python causal_flows/generate_jobs.py --grid_file _params/_grid/causal_nf/base.yaml --format shell --jobs_per_file 20000 --batch_size 500 --wandb_mode disabled
```

Change the batch size if we need to create multiple `.sh` files.

**Note:** For running on cluster with condor, we need to use `--format sub`. This will create `.sub` files that can be used with `condor submit` instead of `.sh` files.

All results are saved into the directory defined in the `base.yaml` file, e.g., for causal flows `causal_nf/`:
```yaml
root_dir: [ hparam_grid_causal_mods/comparison_causal_nf ]
```
### c. Analyzing grid search outputs
To analyze the grid search outcome, run the following scripts:
```bash
python causal_flows/scripts/create_comparison_flows.py hparam_grid_causal_mods/
```
This will print out the best hyperparameters on the console for each dataset.

<a name="run_cgm"></a>
## Running causal generative model
### a. Make param file
First, we need to create a param file for the causal generative model we want to run based on the dataset we want to run it on.
This is also the file where we need to *manually* set the best hyperparameters we found.
One can find the param files we have used for our datasets in `_params/_causal_models/` path.
For instance, for causal flows on treatment-formatted German data, the file `treatment_german_causal_nf.yaml` has the hyperparameter set as:

Note that some default parameters are taken and must be put in the file `default_config.yaml`. This file is read from `causal_flows/causal_nf/configs/`.
This code structure is adopted from the original causal flows [code](https://github.com/psanch21/causal-flows).
### b. Train causal generative model
Train the desired model by running the script `run_causal_estimator.py`. Use the help option for more details:
```bash
python run_causal_estimator.py --help
```
For instance, for the German data, we can train the causal flows model using:
```bash
python run_causal_estimator.py --config_file _params/_causal_models/treatment_german_causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF
```
Again, we can set the checkpoint saving location in the param file, e.g., 
```yaml
root_dir: _models_trained/causal_nf/
```

<a name="run_cf"></a>
## Loading model and generating counterfactuals
Use the command as the following by giving the path to the checkpoint directory and the `pscf` flag:
```python
python run_causal_estimator.py --config_file _params/_causal_models/treatment_german_causal_nf.yaml --wandb_mode disabled --project CAUSAL_NF --load_model _models_trained/causal_nf/treatment_german_dummy --pscf_analysis
```
This execution generates `.csv` files with the counterfactual values for path-specific interventions with the causal normalizing flows. 
The CSV files can then be used to perform all related analysis.

### Analysis and bias mitigation
We can use the generated counterfactuals to perform the fairness auditing of treatment discrimination.
See the `notebooks/analyzer.ipynb` notebook for some representative analysis that can be conducted with the CSV files.

We also provide a notebook `notebooks/mitigation.ipynb` to show how we can perform bias mitigation with the counterfactuals in two stages: i. risk score estimation that is often used to make binary decisions, and ii. obtaining fairer outcomes through fairer treatments.

## Acknowledgements
The code on Causal Normalizing Flows is based on [this repository](https://github.com/psanch21/causal-flows).
