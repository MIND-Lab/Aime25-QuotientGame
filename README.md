# Aime25-QuotientGame
This repository contains the code to replicate the experiments conducted for the paper **"A SHAP Quotient Game for explaining Raman Spectroscopy classification models"**, submitted to [AIME 2025]([url](https://aime25.aimedicine.info/)).

## Dataset
The **dataset** folder contains pickle files with the datasets used for the experiments.

## Model Weights
The **covid** and **pd-ad** folders contain the weights of the models trained using the leave-one-patient-out paradigm in [Keras]([url](https://keras.io/))).

## Utils
The **utils** folder contains utility scripts used in the experiments:

- Dataset.py: Functions for reading and preparing the datasets for experiments.
- GradCam.py: Core functions for applying Grad-CAM.
- quotient_game.py: Implementation of the Quotient Game method.
- kernel_shap.py: A re-implementation of the SHAP method using the [SHAP library]([url](https://shap.readthedocs.io/en/latest/)).
- cnn_models.py: Functions to create the CNN model used for the experiments using [keras python library]([url](https://keras.io/))

## Experiment files
Description of the remaining files in the repository:

- shap.py: Experiments using the [SHAP]([url](https://shap.readthedocs.io/en/latest/)) method.
- lime.py: Experiments using the [LIME]([url](https://github.com/marcotcr/lime)) method.
- qg_experiment.py: Experiments using the Quotient Game method.
- grad_cam.py: Experiments using the Grad-CAM method.

The file **experiments.py** provides an example experiment for each method applied to the PD-AD dataset. It leverages all the available files in the repository.

## Contacts

For any questions or further information, please contact Marco Piazza (m.piazza23@campus.unimib.it) or Enza Messina (enza.messina@unimib.it).
