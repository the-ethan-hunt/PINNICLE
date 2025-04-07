#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:01:06 2025

@author: dhruvgirishapte
This code omits usage of ocean current velocity and sea ice thickness as it 
is difficult to find training data for these quantities. 
"""

import deepxde as dde
from deepxde.backend import jax
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax
import numpy as np

class SeaIceEquationParameter(EquationParameter, Constants):
    """ default parameters for Sea ice equations """
    _EQUATION_TYPE = 'Sea_Ice' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'A']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
            'Pstar': 2.75e4,               # ice strength
            'C': 20,                       # Compressive strength parameter
            'S_to_B': 0.5,                 # Shear to Bulk Viscosity Ratio
            'T': 1e-3,                     # Relaxation Time
            'g': 9.81,
            'rho_a': 1.3                   # Air density
        }

class SeaIce(EquationBase):
    _EQUATION_TYPE = 'Sea_Ice' 
    def __init__(self, parameters=SeaIceEquationParameter()):
        super().__init__(parameters)

    def _pde_jax(self, nn_input_var, nn_output_var):
        """ Residual of sea-ice equations 2D PDEs """
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]
        
        wind_x = self.local_input_var["wind_x"]
        wind_y = self.local_input_var["wind_y"]
        
        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        Aid = self.local_output_var["A"]
        
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)
        
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        A = slice_column(nn_output_var, Aid)

        P = self.Pstar * (2 ** (self.C * (1 - A)))

        Delta = jax.numpy.sqrt(
            u_x**2 + v_y**2 + 4 * (u_x + v_y)**2 + 2 * (0.5 * (u_y + v_x))**2 + 1e-15
        )

        zeta = P / Delta
        eta = zeta / 2.0

        epsilon_xx = u_x
        epsilon_yy = v_y
        epsilon_xy = 0.5 * (u_y + v_x)

        epsilon_dev_xx = epsilon_xx - (1 / 3) * (epsilon_xx + epsilon_yy)
        epsilon_dev_yy = epsilon_yy - (1 / 3) * (epsilon_xx + epsilon_yy)
        epsilon_dev_xy = epsilon_xy

        sigma_xx = 2 * eta * epsilon_dev_xx + zeta * (epsilon_xx + epsilon_yy)
        sigma_yy = 2 * eta * epsilon_dev_yy + zeta * (epsilon_xx + epsilon_yy)
        sigma_xy = 2 * eta * epsilon_dev_xy

        tau = jax.numpy.sqrt(sigma_xx**2 + sigma_yy**2 + sigma_xy**2)
        sigma_yield = P
        scaling_factor = jax.numpy.where(tau > sigma_yield, sigma_yield / tau, 1.0)
        sigma_xx *= scaling_factor
        sigma_yy *= scaling_factor
        sigma_xy *= scaling_factor

        div_sigma_x = jacobian(sigma_xx, nn_input_var, i=xid, val=1) + jacobian(sigma_xy, nn_input_var, i=yid, val=1)
        div_sigma_y = jacobian(sigma_xy, nn_input_var, i=xid, val=1) + jacobian(sigma_yy, nn_input_var, i=yid, val=1)

        C_a = 0.002  # Air-ice drag coefficient

        Ua = jax.numpy.array([wind_x, wind_y])
        Ui = jax.numpy.array([u, v])

        Ua_minus_Ui = Ua - Ui
        tau_ax = self.rho_a * C_a * jax.numpy.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[0]
        tau_ay = self.rho_a * C_a * jax.numpy.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[1]

        f1 = div_sigma_x - tau_ax
        f2 = div_sigma_y - tau_ay

        return [f1, f2]
