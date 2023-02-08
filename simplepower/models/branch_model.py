from typing import Optional
from math import sqrt
from sympy import symbols, diff, lambdify, cos
import numpy as np
import numdifftools as nd 
from dataclasses import dataclass
from copy import deepcopy

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
    is_pu: Optional[bool] = False
    
    def __post_init__(self) -> None: 
        if not self.is_pu:
            self.g_1 = self.g_1 * 1e-6 # convert from uS to S
            self.g_2 = self.g_2 * 1e-6 
            self.b_1 = self.b_1 * 1e-6 
            self.b_2 = self.b_2 * 1e-6  

        self.Z_base = self.V_base_kV**2 / self.S_base_mva 
        self.Y_base = 1/self.Z_base 
        self.g_l = self.r_l / (self.r_l**2 + self.x_l**2)
        self.b_l = self.x_l / (self.r_l**2 + self.x_l**2)

        self.y_series = self.g_l - 1j*self.b_l 
        self.y_1_shunt = self.g_1 + 1j*self.b_1 
        self.y_2_shunt = self.g_2 + 1j*self.b_2 

    def convert_to_pu(self): 
        return BranchDataClass(self.S_base_mva, self.V_base_kV, self.r_l/self.Z_base, self.x_l/self.Z_base, 
                               self.g_1/self.Y_base, self.b_1/self.Y_base, self.g_2/self.Y_base, self.b_2/self.Y_base, is_pu=True)

    def convert_from_pu(self): 
        return BranchDataClass(self.S_base_mva, self.V_base_kV, self.r_l*self.Z_base, self.x_l*self.Z_base, 
                               self.g_1*self.Y_base, self.b_1*self.Y_base, self.g_2*self.Y_base, self.b_2*self.Y_base, is_pu=False)
    
    def change_base(self, S_new_mva: float, V_new_kV: float): 
        """Changes all pu bases from S_base_mva, V_base_kV to a new set of S_new_mva, V_new_kV. NOTE: Assumes class already in pu."""
        Z_b_new = V_new_kV**2/S_new_mva 
        return BranchDataClass(S_new_mva, V_new_kV, self.r_l*self.Z_base/Z_b_new, self.x_l*self.Z_base/Z_b_new, 
                               self.g_1*self.Y_base*Z_b_new, self.b_1*self.Y_base*Z_b_new, 
                               self.g_2*self.Y_base*Z_b_new, self.b_2*self.Y_base*Z_b_new, is_pu=True)


class TrafoDataClass(BranchDataClass):
    """Assume the Kundur transformer model. """ 
    def __init__(self, S_base_mva: float, V_n_hv: float, V_n_lv: float, V_SCH: float, P_Cu: float, I_E: float, P_Fe: float, 
                 tap_change: Optional[float] = 0.01, tap_min: Optional[int] = -7, tap_max: Optional[int] = 7, 
                 r_leak_hv: Optional[float] = 0.5, x_leak_hv: Optional[float] = 0.5): 
        """
        Dataclass for transformer modelAll values is specified in pu 

        Attributes 
        ----------

        If defined in real units, call the "convert_to_pu" method to get the pu class representation. 
        S_n: Rated power [MVA] (also base power) 
        V_n_hv: Rated voltage [kV] on the HV side 
        V_n_lv: Rated voltage [kV] on the LV side 
        V_SCH: The voltage that causes nominal current with the transformer short-circuited [pu] 
        P_Cu: Copper losses during nominal operating point [pu] 
        I_E: No-load current [pu] 
        P_Fe: No-load power losses [pu] 
        tap_change: pu change of the voltage ratio per tap 
        tap_min: minimum tap position 
        tap_max: maximum tap position
        r_leak_hv: How much leakage resistance there is at the hv side. The rest (1-r_leak_hv) is at the lv side. 
        x_leak_hv: How much leakage reactance there is at the hv side. The rest (1-r_leak_hv) is at the lv side. 
        """
        self.n_ratio = 1.0 # TODO: Calculate this based on the tap changer
        self.c_ratio = 1/self.n_ratio
        self.Z_base_hv = V_n_hv**2/S_base_mva
        self.Z_base_lv = V_n_lv**2/S_base_mva 
        self.Y_base_hv = self.Z_base_hv**-1
        self.Y_base_lv = self.Z_base_lv**-1
        self.r_leak_hv = r_leak_hv 
        self.x_leak_hv = x_leak_hv

        self.Z_T = V_SCH 
        self.R_T = P_Cu
        self.X_T = sqrt(self.Z_T**2 - self.R_T**2) 

        R_hv = self.R_T * self.r_leak_hv * self.Z_base_hv 
        R_lv = self.R_T * (1-self.r_leak_hv) * self.Z_base_lv 
        X_hv = self.X_T * self.x_leak_hv * self.Z_base_hv 
        X_lv = self.X_T * (1-self.x_leak_hv) * self.Z_base_lv 
        Z_hv = R_hv + 1j*X_hv
        Z_lv = R_lv + 1j*X_lv
        Z_e = self.n_ratio**2 * (Z_hv + Z_lv)
        Y_e = Z_e**-1

        R_1 = Z_e.real * self.n_ratio
        X_1 = abs(Z_e.imag * self.n_ratio )
        
        G_2 = Y_e.real * self.c_ratio * (self.c_ratio - 1)
        B_2 = abs(Y_e.imag * self.c_ratio * (self.c_ratio - 1))
        
        G_3 = Y_e.real *  (1 - self.c_ratio)
        B_3 = abs(Y_e.imag *  (1 - self.c_ratio))

        self.G_Fe = P_Fe * self.Z_base_hv
        self.B_mu = sqrt((I_E*self.Z_base_hv)**2 - self.G_Fe**2)
        self.V_n_hv = V_n_hv 
        self.V_n_lv = V_n_lv
        self.tap_change = tap_change
        self.tap_min = tap_min 
        self.tap_max = tap_max 


        super().__init__(S_base_mva, self.V_n_hv, R_1, X_1, G_2+self.G_Fe, B_2+self.B_mu, 
                         G_3, B_3, is_pu=False)
        
    def change_base(self, S_new_mva: float, V_new_kV: float): 
        """Changes all pu bases from S_base_mva, V_base_kV to a new set of S_new_mva, V_new_kV. NOTE: Assumes class already in pu."""
        Z_b_new = V_new_kV**2/S_new_mva 
        return BranchDataClass(S_new_mva, V_new_kV, self.r_l*self.Z_base_hv/Z_b_new, self.x_l*self.Z_base_hv/Z_b_new, 
                               self.g_1*self.Y_base_hv*Z_b_new, self.b_1*self.Y_base_hv*Z_b_new, 
                               self.g_2*self.Y_base_lv*Z_b_new, self.b_2*self.Y_base_lv*Z_b_new, is_pu=True)


class LineDataClass(BranchDataClass): 
    def __init__(self, S_base_mva: float, V_base_kV: float, r_ohm_per_km: float, x_ohm_per_km: float, 
                 c_uf_per_km: float, length: float, is_pu: bool, f_nom=50): 
        """
        Dataclass for line data. If defined in real units, call the "convert_to_pu" method to get the pu class representation. 

        Attributes 
        ---------- 
        S_n: Rated power [MVA] (also base power) 
        V_base_kV: Rated voltage [kV] 
        r_ohm_per_km: Line/series resistance per km 
        x_ohm_per_km: Line/series reactance per km 
        c_uf_per_km: Shunt capacitance in micro Farad per km 
        length: Total line length 
        f_nom: System frequency
        """
        self.R_line = r_ohm_per_km * length 
        self.X_line = x_ohm_per_km * length  
        self.B_shunt = c_uf_per_km * length * 2*np.pi*f_nom 
        super().__init__(S_base_mva, V_base_kV, self.R_line, self.X_line, 0, self.B_shunt/2, 0, self.B_shunt/2, is_pu=is_pu)


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

