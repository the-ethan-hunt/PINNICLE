#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:33:13 2025

@author: dhruvgirishapte
"""

import pinnicle as pinn
import os
import numpy as np
import deepxde as dde
from preprocess_sea_ice_data import SeaIceDataPreparer

# Set default configurations for DeepXDE
dde.config.set_default_float("float64")
dde.config.disable_xla_jit()
dde.config.set_random_seed(1234)

# General parameters for the experiment
hp = {}
hp["epochs"] = 10# 500000  # Fewer epochs for testing initially
hp["learning_rate"] = 0.001
hp["loss_function"] = "MSE"  # Mean Squared Error
hp["save_path"] = "./Models/SeaIce_PINN/"
hp["is_save"] = True
hp["is_plot"] = True

# Neural Network structure
hp["activation"] = "tanh"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 30  # Increase neurons for a potentially more expressive network
hp["num_layers"] = 4

# Domain (this would need to be defined according to your physical region of interest)
hp["shapefile"] = "sea_ice_grid.exp"
hp["num_collocation_points"] = 5000  # Adjust based on computational resources

# Physics-informed equations (adapted for sea-ice dynamics)
Sea_Ice = {}
Sea_Ice["scalar_variables"] = {
        'Pstar': 2.75e4,               # ice strength
        'C': 20,                       # Compressive strength parameter
        'S_to_B': 0.5,                  # Shear to Bulk Viscosity Ratio
        'T': 1e-3,                       # Relaxation Time
        'g': 9.81,
        'rhoi': 900,                     # Sea-ice density
        'rho_w': 1025,                  # Ocean water density
        'rho_a': 1.3                    # Air density
}

hp["equations"] = {"Sea_Ice": Sea_Ice}

# Process and load the data
file_path = "/Users/dhruvgirishapte/Downloads/dataM1/processing/CNN_data_M1.mat"
data_preparer = SeaIceDataPreparer(file_path)
prepared_data = data_preparer.prepare_data()

# Training, validation, and testing datasets
X_train, Y_train = prepared_data["X_tr"], prepared_data["Y_tr"]
X_val, Y_val = prepared_data["X_va"], prepared_data["Y_va"]
X_test, Y_test = prepared_data["X_te"], prepared_data["Y_te"]

# Data input/output structure for the model
sea_ice_data = {
    "data_size": {
        "u": X_train.shape[0],  # Use the training size dynamically
        "v": X_train.shape[0],  # Assuming U and V components are used
    },
    "data_path": file_path,  # This isn't directly used since we preprocess, but it keeps consistency
}
hp["data"] = {
    "Sea_Ice": {
        "u": X_test[0],  # U-component of wind
        "v": X_test[1],  # V-component of wind
        "sivelu_prev": X_test[2],  # Previous timestep U-component of sea ice velocity
        "sivelv_prev": X_test[3],  # Previous timestep V-component of sea ice velocity
        "siconc_prev": X_test[4],  # Previous timestep sea ice concentration
    }
}


# Additional loss function (customized for velocity if necessary)
vel_loss = {
    "name": "Velocity Logarithmic Loss",
    "function": "VEL_LOG",
    "weight": 1.0e-5,  # Adjust weight as needed
}
hp["additional_loss"] = {"vel": vel_loss}

# Create and compile the PINN experiment
experiment = pinn.PINN(hp)
print(experiment.params)
experiment.compile()

# Train the PINN model
experiment.train(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)

# Evaluate the model and visualize results
experiment.plot_predictions(
    X_ref=X_test,  # Reference test inputs
    sol_ref=Y_test,  # Reference test outputs
)
