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

    model1_num = BranchModel(branch_data)
    model1_sym = BranchModel(branch_data, mode='sym')

    model2_num = BranchModel(branch_data2)
    model2_sym = BranchModel(branch_data2, mode='sym')

    # Vals for case 1
    V1 = 1.204158 
    V2 = 1.0 
    d1 = 4.764 * np.pi/180 
    d2 = 0.0 
    X = np.array([d1, d2, V1, V2])
    P_loss_pre = 0.05 
    Q_loss_pre = 0.15 

    def test_loss_calc_model1(self): 
        P_loss1, Q_loss1 = self.model1_num.y(np.array([self.d1, self.d2, self.V1, self.V2]))
        P_loss2, Q_loss2 = self.model1_sym.y(np.array([self.d1, self.d2, self.V1, self.V2]))
        print("Model 1: ")
        print(f"P_loss1 = {P_loss1}, Q_loss1 = {Q_loss1}, P_loss_pre = {self.P_loss_pre}, Q_loss_pre = {self.Q_loss_pre}")
        print(f"P_loss2 = {P_loss2}, Q_loss2 = {Q_loss2}, P_loss_pre = {self.P_loss_pre}, Q_loss_pre = {self.Q_loss_pre}")
        print("")

    def test_loss_calc_mode2(self): 
        P_loss1, Q_loss1 = self.model2_num.y(np.array([self.d1, self.d2, self.V1, self.V2]))
        P_loss2, Q_loss2 = self.model2_sym.y(np.array([self.d1, self.d2, self.V1, self.V2]))
        print("Model 2:")
        print(f"P_loss1 = {P_loss1}, Q_loss1 = {Q_loss1}, P_loss_pre = {self.P_loss_pre}, Q_loss_pre = {self.Q_loss_pre}")
        print(f"P_loss2 = {P_loss2}, Q_loss2 = {Q_loss2}, P_loss_pre = {self.P_loss_pre}, Q_loss_pre = {self.Q_loss_pre}")
        print("")

    def test_loss_grad_model1_sym(self): 
        dy_dx_sym = self.model1_sym.dy_dx(self.X)
        for P_calc in dy_dx_sym[0]: 
            self.assertIsNotNone(P_calc)
        for Q_calc in dy_dx_sym[1]: 
            self.assertIsNotNone(Q_calc)

    def test_loss_grad_model1_num(self):
        dy_dx_num = self.model1_num.dy_dx(self.X)
        for P_calc in dy_dx_num[0]: 
            self.assertIsNotNone(P_calc)
        for Q_calc in dy_dx_num[1]: 
            self.assertIsNotNone(Q_calc)

    def test_loss_grad_model2_sym(self): 
        dy_dx_sym = self.model2_sym.dy_dx(self.X)
        for P_calc in dy_dx_sym[0]: 
            self.assertIsNotNone(P_calc)
        for Q_calc in dy_dx_sym[1]: 
            self.assertIsNotNone(Q_calc)

    def test_loss_grad_model2_num(self):
        dy_dx_num = self.model2_num.dy_dx(self.X)
        for P_calc in dy_dx_num[0]: 
            self.assertIsNotNone(P_calc)
        for Q_calc in dy_dx_num[1]: 
            self.assertIsNotNone(Q_calc)


if __name__ == "__main__": 
    unittest.main() 