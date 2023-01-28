import unittest 
import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from simplepower.models.branch_model import BranchDataClass, BranchModel


class TestDataClass(unittest.TestCase): 
    branch_data = BranchDataClass(S_base_mva=100, V_base_kV=132, r_l=10, x_l=40) # A basic model without shunts 
    branch_data2 = BranchDataClass(S_base_mva=100, V_base_kV=132, r_l=10, x_l=40, g_1=0.01, b_1=0.3, g_2=0.01, b_2=0.3 ) # A basic model without shunts 

    def test_data_class_basic(self): 
        new_model = self.branch_data.convert_to_pu()
        other_model = new_model.convert_from_pu() 
        self.assertEqual(self.branch_data, other_model)

    def test_data_class_2(self): 
        new_model = self.branch_data2.convert_to_pu()
        other_model = new_model.convert_from_pu() 
        self.assertEqual(self.branch_data2, other_model)


class TestBranchModel(unittest.TestCase): 
    branch_data = BranchDataClass(S_base_mva=100, V_base_kV=132, r_l=0.1, x_l=0.3) # A basic model without shunts 
    branch_data2 = BranchDataClass(S_base_mva=100, V_base_kV=132, r_l=10, x_l=40, g_1=0.01, b_1=0.3, g_2=0.01, b_2=0.3).convert_to_pu() # A basic model without shunts 

    model1 = BranchModel(branch_data)
    model2 = BranchModel(branch_data2)

    # Vals for case 1
    V1 = 1.204158 
    V2 = 1.0 
    d1 = 4.764 * np.pi/180 
    d2 = 0.0 
    P_loss_pre = 0.05 
    Q_loss_pre = 0.15 

    def calc_dy_dx_numerically(self): 
        d_delta = 1e-3
        d_V = 1e-3
        mod_x = np.array([d_delta, d_delta, d_V, d_V])
        X_nom = np.array([self.d1, self.d2, self.V1, self.V2])

        dP_losses = np.zeros(4, dtype=np.float64)
        dQ_losses = np.zeros(4, dtype=np.float64)
        for i in range(4): #For the P_losses:
            mod = np.zeros(4) 
            mod[i] += mod_x[i]

            X_low = X_nom - mod
            X_high = X_nom + mod
            P_loss_low, Q_loss_low = self.model1.y(X_low)
            P_loss_high, Q_loss_high = self.model1.y(X_high)
            dP_losses[i] = (P_loss_high - P_loss_low)/(2*mod_x[i])
            dQ_losses[i] = (Q_loss_high - Q_loss_low)/(2*mod_x[i])

        self.dy_dx_num = np.array([dP_losses, dQ_losses])

    def test_loss_calc(self): 
        P_loss, Q_loss = self.model1.y(np.array([self.d1, self.d2, self.V1, self.V2]))
        self.assertAlmostEqual(P_loss, self.P_loss_pre, places=4)
        self.assertAlmostEqual(Q_loss, self.Q_loss_pre, places=4)

    def test_loss_grad(self): 
        self.calc_dy_dx_numerically()
        dy_dx = self.model1.dy_dx(np.array([self.d1, self.d2, self.V1, self.V2]))
        for P_calc, P_est in zip(self.dy_dx_num[0], dy_dx[0]): 
            self.assertAlmostEqual(P_calc, P_est, places=3)
        for Q_calc, Q_est in zip(self.dy_dx_num[1], dy_dx[1]): 
            self.assertAlmostEqual(Q_calc, Q_est, places=3)


if __name__ == "__main__": 
    unittest.main() 