from typing import Optional
from math import sqrt
from sympy import symbols, diff, lambdify, cos
import numpy as np
import numdifftools as nd 
from dataclasses import dataclass

@dataclass
class BranchDataClass: 
    """Data class for any branch model, including transmission line, cable, or trafo. \n
    NOTE: Missing functionallity for phase-shifters. \n 
    NOTE: All values can be specified in either pu or in the following units. \n 
    If defined in real units, call the "convert_to_pu" method to get the pu class representation. \n
    S_base_mva: Base power [MVA] (for converting between ohms/Simens to pu) \n
    V_base_kV: Base voltage [kV]  (for converting between ohms/Simens to pu) \n 
    r_l: Series branch resistance [Ohm] \n 
    x_l: Series branch reactance [Ohm] \n 
    g_1: Shunt conductance at port 1 [uS] \n 
    b_1: Shunt conductance at port 1 [uS] \n 
    g_2: Shunt conductance at port 2 [uS] \n 
    b_2: Shunt conductance at port 2 [uS] \n    
    """
    S_base_mva: float 
    V_base_kV: float
    r_l: float 
    x_l: float 
    g_1: Optional[float] = 0.0
    b_1: Optional[float] = 0.0
    g_2: Optional[float] = 0.0
    b_2: Optional[float] = 0.0
    
    def __post_init__(self) -> None: 
        self.g_1 = self.g_1 * 1e-6 # convert from uS to S
        self.g_2 = self.g_2 * 1e-6 
        self.b_1 = self.b_1 * 1e-6 
        self.b_2 = self.b_2 * 1e-6  

        self.Z_base = self.V_base_kV**2 / self.S_base_mva 
        self.Y_base = 1/self.Z_base 
        self.g_l = self.r_l / (self.r_l**2 + self.x_l**2)
        self.b_l = self.x_l / (self.r_l**2 + self.x_l**2)

    def convert_to_pu(self): 
        return BranchDataClass(self.S_base_mva, self.V_base_kV, self.r_l/self.Z_base, self.x_l/self.Z_base, 
                               self.g_1/self.Y_base*1e6, self.b_1/self.Y_base*1e6, self.g_2/self.Y_base*1e6, self.b_2/self.Y_base*1e6)

    def convert_from_pu(self): 
        return BranchDataClass(self.S_base_mva, self.V_base_kV, self.r_l*self.Z_base, self.x_l*self.Z_base, 
                               self.g_1*self.Y_base*1e6, self.b_1*self.Y_base*1e6, self.g_2*self.Y_base*1e6, self.b_2*self.Y_base*1e6)


class TrafoDataClass(BranchDataClass): 
    def __init__(self, S_base_mva: float, V_n_hv: float, V_n_lv: float, V_SCH: float, P_Cu: float, I_E: float, P_Fe: float, 
                 tap_change: Optional[float] = 0.01, tap_min: Optional[int] = -7, tap_max: Optional[int] = 7): 
        """
        NOTE: All values can be specified in either pu or in the following units. \n 
        If defined in real units, call the "convert_to_pu" method to get the pu class representation. \n
        S_n: Rated power [MVA] (also base power) \n 
        V_n_hv: Rated voltage [kV] on the HV side \n 
        V_n_lv: Rated voltage [kV] on the LV side \n 
        V_SCH: The voltage that causes nominal current with the transformer short-circuited [pu] \n  
        P_Cu: Copper losses during nominal operating point [pu] \n 
        I_E: No-load current [pu] \n 
        P_Fe: No-load power losses [pu] \n 
        tap_change: pu change of the voltage ratio per tap \n 
        tap_min: minimum tap position \n 
        tap_max: maximum tap position \n
        """
        self.Z_T = V_SCH 
        self.R_T = P_Cu 
        self.X_T = sqrt(self.Z_T**2 - self.R_T**2) 
        self.G_Fe = P_Fe
        self.B_mu = sqrt(I_E**2 - self.G_Fe)
        self.V_n_hv = V_n_hv 
        self.V_n_lv = V_n_lv
        self.tap_change = tap_change
        self.tap_min = tap_min 
        self.tap_max = tap_max 
        super().__init__(S_base_mva, V_n_hv, self.R_T, self.X_T, self.G_Fe/2, self.B_mu/2, self.G_Fe/2, self.B_mu/2)


class BranchModel: 
    def __init__(self, branch_data: BranchDataClass, mode: Optional[str]="num"): 
        """
        Defines the equations for power losses. 

        Attributes 
        ----------
        branch_data: BranchDataClass, either a dataclass from line or trafo model 
        mode: Either symbolic or numeric, decides if equations are sympy-symbolic or python numerical. Options: ('num', 'sym')

        Internal attributes 
        ---------
        model.x_names: A tuple of the state names required for the model to work 
        model.y_names: A tuple of the algebraic variable names calculated in the model

        Internal functions: 
        ----------
        model.y -> [P_loss, Q_loss], returns line power losses \n 
        model.dy_dx -> Grad(model.y w.r.t. model.x), shape = (N_y, N_x) \n
        When inputing anything to the y or dy_dx functions, the X = ndarray([d_1, d_2, V_1, V_2])
        """
        self.md = branch_data
        self.x_names = ("d_1", "d_2", "V_1", "V_2") # Externally obtained states
        self.y_names = ("P_loss_pu", "Q_loss_pu") # Alg vars noted y
        self.mode = mode
        if mode == 'sym': 
            self._define_symbolic()
        else: 
            self._define_numerical()

    def _define_numerical(self): 
        P_loss = lambda X: X[2]**2*self.md.g_1 + X[3]**2*self.md.g_2 + self.md.g_l*(X[2]**2 + X[3]**2 -2*X[2]*X[3]*cos(X[0]-X[1]))
        Q_loss = lambda X: X[2]**2*self.md.b_1 + X[3]**2*self.md.b_2 + self.md.b_l*(X[2]**2 + X[3]**2 -2*X[2]*X[3]*cos(X[0]-X[1]))
        self.y = lambda X: np.array([P_loss(X), Q_loss(X)], dtype=float)
        self.dy_dx = nd.Gradient(self.y)

    def _define_symbolic(self): 
        V1, V2, d1, d2 = symbols("V1 V2 d1 d2")
        X_sym = (d1, d2, V1, V2)

        P_loss = V1**2*self.md.g_1 + V2**2*self.md.g_2 + self.md.g_l*(V1**2 + V2**2 -2*V1*V2*cos(d1-d2))
        Q_loss = V1**2*self.md.b_1 + V2**2*self.md.b_2 + self.md.b_l*(V1**2 + V2**2 -2*V1*V2*cos(d1-d2))

        dP_loss_dV1 = diff(P_loss, V1)
        dP_loss_dV2 = diff(P_loss, V2)
        dP_loss_dd1 = diff(P_loss, d1)
        dP_loss_dd2 = diff(P_loss, d2)

        dQ_loss_dV1 = diff(Q_loss, V1)
        dQ_loss_dV2 = diff(Q_loss, V2)
        dQ_loss_dd1 = diff(Q_loss, d1)
        dQ_loss_dd2 = diff(Q_loss, d2)


        self.y = lambda X: np.array([lambdify(X_sym, P_loss)(*X), 
                                     lambdify(X_sym, Q_loss)(*X)])
        self.dy_dx = lambda X: np.array([[lambdify(X_sym, dP_loss_dd1)(*X), 
                                          lambdify(X_sym, dP_loss_dd2)(*X), 
                                          lambdify(X_sym, dP_loss_dV1)(*X), 
                                          lambdify(X_sym, dP_loss_dV2)(*X)], 
                                         [lambdify(X_sym, dQ_loss_dd1)(*X),
                                          lambdify(X_sym, dQ_loss_dd2)(*X),
                                          lambdify(X_sym, dQ_loss_dV1)(*X),
                                          lambdify(X_sym, dQ_loss_dV2)(*X)] 
                                          ])

    def __repr__(self): 
        return self.md.__repr__()
    
    def __str__(self): 
        return self.md.__str__()

