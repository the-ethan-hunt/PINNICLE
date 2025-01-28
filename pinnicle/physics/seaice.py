import deepxde as dde
from deepxde.backend import jax
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax
import numpy as np
class Sea_IceEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA on ice shelves
    """
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
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'Pstar': 2.75e4,               # ice strength
                'C': 20,                       # Compressive strength parameter
                'S_to_B': 0.5,                  # Shear to Bulk Viscosity Ratio
                'T': 1e-3,                       # Relaxation Time
                'g': 9.81,
                'rhoi': 900,                     # Sea-ice density
                'rho_w': 1025,                  # Ocean water density
                'rho_a': 1.3                    # Air density
                }
class Sea_Ice(EquationBase): #{{{
    _EQUATION_TYPE = 'Sea_Ice' 
    def __init__(self, parameters=Sea_IceEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of sea ice 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]
        
        wind_x = self.local_input_var["wind_x"]
        wind_y = self.local_input_var["wind_y"]
        #ocean_x = self.local_input_var["ocean_x"]
        #ocean_y = self.local_input_var["ocean_y"]
        
        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        #Hid = self.local_output_var["H"]
        Aid = self.local_output_var["A"]

        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)

        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        #H = slice_column(nn_output_var, Hid)
        A = slice_column(nn_output_var, Aid)

        P = self.Pstar * (2 **(self.C * (1 - A))) # Ice strength
        Delta = np.sqrt(u_x**2 + v_y**2 + 4 * (u_x + v_y)**2 + 2 * (0.5 * (u_y + v_x))**2 + 1e-15 ) # Measure of deformation rate
        zeta = P/Delta
        eta = zeta/2.0
        
        # Strain rate tensor
        
        epsilon_xx = u_x
        epsilon_yy = v_y
        epsilon_xy = 0.5*(u_y + v_x)
        
        # Deviatoric strain tensor
        
        epsilon_dev_xx = epsilon_xx - (1/3)*(epsilon_xx + epsilon_yy)
        epsilon_dev_yy = epsilon_yy - (1/3)*(epsilon_xx + epsilon_yy)
        epsilon_dev_xy = epsilon_xy
        
        # Stress tensor components
        
        sigma_xx = 2*eta*epsilon_dev_xx + zeta*(epsilon_xx + epsilon_yy)
        sigma_yy = 2*eta*epsilon_dev_yy + zeta*(epsilon_xx + epsilon_yy)
        sigma_xy = 2*eta*epsilon_dev_xy
        
        # apply plasticity
        
        tau = jax.numpy.sqrt(sigma_xx**2 + sigma_yy**2 + sigma_xy**2)
        sigma_yield = P
        scaling_factor = jax.numpy.where(tau > sigma_yield, sigma_yield / tau, 1.0)
        sigma_xx *= scaling_factor
        sigma_yy *= scaling_factor
        sigma_xy *= scaling_factor
        
        # Compute stress divergence
        div_sigma_x = jacobian(sigma_xx, nn_input_var, i=xid) + jacobian(sigma_xy, nn_input_var, i=yid)
        div_sigma_y = jacobian(sigma_xy, nn_input_var, i=xid) + jacobian(sigma_yy, nn_input_var, i=yid)
        C_a = 0.014  # Air-ice drag coefficient
        #C_o = 0.005  # Ice-ocean drag coefficient
        
        # Velocities
        Ua = np.array([wind_x, wind_y])  # Atmospheric wind velocity (m/s)
        #Uo = np.array([ocean_x, ocean_y])  # Ocean current velocity (m/s)
        Ui = np.array([u, v])  # Ice velocity (m/s)
        Ua_minus_Ui = Ua - Ui
        
        # Atmospheric stress components
        tau_ax = self.rho_a * C_a * jax.np.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[0]
        tau_ay = self.rho_a * C_a * jax.np.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[1]
        
        # Oceanic stress components
        #tau_ox = self.rhoi * C_o * np.linalg.norm(Uo - Ui) * (Uo[0] - Ui[0])
        #tau_oy = self.rhoi * C_o * np.linalg.norm(Uo - Ui) * (Uo[1] - Ui[1])
        
        # Compute driving forces
        #sea_surface_tilt_x = self.rhow * self.g * H * jacobian(H, nn_input_var, i=xid)
        #sea_surface_tilt_y = self.rhow * self.g * H * jacobian(H, nn_input_var, i=yid)
        
        # Residuals
        f1 = div_sigma_x - tau_ax #- tau_ox - sea_surface_tilt_x
        f2 = div_sigma_y - tau_ay #- tau_oy - sea_surface_tilt_y
        
        return [f1, f2]

        
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of sea-ice equations 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # Get input and output variable IDs
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]
        
        wind_x = self.local_input_var["wind_x"]
        wind_y = self.local_input_var["wind_y"]
        #ocean_x = self.local_input_var["ocean_x"]
        #ocean_y = self.local_input_var["ocean_y"]
        
        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        #Hid = self.local_output_var["H"]
        Aid = self.local_output_var["A"]
        
        # Derivatives using JAX jacobian
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)
        
        # Slicing variables
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        #H = slice_column(nn_output_var, Hid)
        A = slice_column(nn_output_var, Aid)

        # Ice strength
        P = self.Pstar * (2 ** (self.C * (1 - A)))

        # Measure of deformation rate
        Delta = jax.numpy.sqrt(
            u_x**2 + v_y**2 + 4 * (u_x + v_y)**2 + 2 * (0.5 * (u_y + v_x))**2 + 1e-15
            )

        zeta = P / Delta
        eta = zeta / 2.0

        # Strain rate tensor
        epsilon_xx = u_x
        epsilon_yy = v_y
        epsilon_xy = 0.5 * (u_y + v_x)

        # Deviatoric strain tensor
        epsilon_dev_xx = epsilon_xx - (1 / 3) * (epsilon_xx + epsilon_yy)
        epsilon_dev_yy = epsilon_yy - (1 / 3) * (epsilon_xx + epsilon_yy)
        epsilon_dev_xy = epsilon_xy

        # Stress tensor components
        sigma_xx = 2 * eta * epsilon_dev_xx + zeta * (epsilon_xx + epsilon_yy)
        sigma_yy = 2 * eta * epsilon_dev_yy + zeta * (epsilon_xx + epsilon_yy)
        sigma_xy = 2 * eta * epsilon_dev_xy

        # Apply plasticity
        tau = jax.numpy.sqrt(sigma_xx**2 + sigma_yy**2 + sigma_xy**2)
        sigma_yield = P
        scaling_factor = jax.numpy.where(tau > sigma_yield, sigma_yield / tau, 1.0)
        sigma_xx *= scaling_factor
        sigma_yy *= scaling_factor
        sigma_xy *= scaling_factor

        # Compute stress divergence
        div_sigma_x = jacobian(sigma_xx, nn_input_var, i=xid, val=1) + jacobian(sigma_xy, nn_input_var, i=yid, val=1)
        div_sigma_y = jacobian(sigma_xy, nn_input_var, i=xid, val=1) + jacobian(sigma_yy, nn_input_var, i=yid, val=1)

        # Air and ocean drag coefficients
        C_a = 0.014  # Air-ice drag coefficient
        #C_o = 0.005  # Ice-ocean drag coefficient

        # Atmospheric and oceanic velocities
        Ua = jax.numpy.array([wind_x, wind_y])  # Atmospheric wind velocity (m/s)
        #Uo = jax.numpy.array([ocean_x, ocean_y])  # Ocean current velocity (m/s)
        Ui = jax.numpy.array([u, v])  # Ice velocity (m/s)

        # Atmospheric stress components
        Ua_minus_Ui = Ua - Ui
        tau_ax = self.rho_a * C_a * jax.numpy.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[0]
        tau_ay = self.rho_a * C_a * jax.numpy.linalg.norm(Ua_minus_Ui) * Ua_minus_Ui[1]

        # Oceanic stress components
        #Uo_minus_Ui = Uo - Ui
        #tau_ox = self.rhoi * C_o * jax.numpy.linalg.norm(Uo_minus_Ui) * Uo_minus_Ui[0]
        #tau_oy = self.rhoi * C_o * jax.numpy.linalg.norm(Uo_minus_Ui) * Uo_minus_Ui[1]

        # Compute driving forces
        #sea_surface_tilt_x = self.rhow * self.g * H * jacobian(H, nn_input_var, i=xid, val=1)
        #sea_surface_tilt_y = self.rhow * self.g * H * jacobian(H, nn_input_var, i=yid, val=1)

        # Residuals
        f1 = div_sigma_x - tau_ax #- tau_ox - sea_surface_tilt_x
        f2 = div_sigma_y - tau_ay #- tau_oy - sea_surface_tilt_y

        return [f1, f2]
