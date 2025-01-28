#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:00:13 2025

@author: dhruvgirishapte
"""

import numpy as np
import h5py


class SeaIceDataPreparer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        """Load the .mat file using h5py."""
        with h5py.File(self.file_path, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}
        return data

    def prepare_data(self):
        """Prepare the data for training, validation, and testing."""
        data = {}

        # Extract training data
        data["X_tr"] = self._process_inputs(self.data["X_tr"])
        data["Y_tr"] = self._process_outputs(self.data["Y_tr"])

        # Extract validation data
        data["X_va"] = self._process_inputs(self.data["X_va"])
        data["Y_va"] = self._process_outputs(self.data["Y_va"])

        # Extract testing data
        data["X_te"] = self._process_inputs(self.data["X_te"])
        data["Y_te"] = self._process_outputs(self.data["Y_te"])

        return data

    def _process_inputs(self, X):
        """
        Process the 4D input array into a flattened or structured format
        suitable for PINN training.
        """
        # Transpose axes since MATLAB arrays are stored column-major
        X = X.transpose(3, 2, 1, 0)

        # Flatten the spatial dimensions (if necessary)
        num_samples = X.shape[0]
        spatial_dims = np.prod(X.shape[1:3])  # Combine spatial dimensions
        num_features = X.shape[3]

        # Reshape into [num_samples * spatial_points, features]
        X_flat = X.reshape(num_samples * spatial_dims, num_features)

        return X_flat

    def _process_outputs(self, Y):
        """
        Process the 4D output array into a flattened format for PINN training.
        """
        # Transpose axes since MATLAB arrays are stored column-major
        Y = Y.transpose(3, 2, 1, 0)

        # Flatten the spatial dimensions (if necessary)
        num_samples = Y.shape[0]
        spatial_dims = np.prod(Y.shape[1:3])  # Combine spatial dimensions
        num_features = Y.shape[3]

        # Reshape into [num_samples * spatial_points, features]
        Y_flat = Y.reshape(num_samples * spatial_dims, num_features)

        return Y_flat


# Example usage:
file_path = "/Users/dhruvgirishapte/Downloads/dataM1/processing/CNN_data_M1.mat"
data_preparer = SeaIceDataPreparer(file_path)
prepared_data = data_preparer.prepare_data()

# Access prepared data:
X_train = prepared_data["X_tr"]
Y_train = prepared_data["Y_tr"]
X_val = prepared_data["X_va"]
Y_val = prepared_data["Y_va"]
X_test = prepared_data["X_te"]
Y_test = prepared_data["Y_te"]

# Print shapes to verify:
print("Training Inputs Shape:", X_train.shape)
print("Training Outputs Shape:", Y_train.shape)
print("Validation Inputs Shape:", X_val.shape)
print("Validation Outputs Shape:", Y_val.shape)
print("Testing Inputs Shape:", X_test.shape)
print("Testing Outputs Shape:", Y_test.shape)

