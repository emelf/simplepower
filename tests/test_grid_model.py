import unittest 
import numpy as np
import os
import sys
import inspect
import pandas as pd 

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from simplepower.models import GridModel
from simplepower.models import GridDataClass, FileType

grid_data_1 = GridDataClass("tests/test_grid_1.xlsx", filetype=FileType.Excel, f_nom=50)
grid_model_1 = GridModel(grid_data_1) 

grid_data_2 = GridDataClass("tests/test_grid_2.xlsx", filetype=FileType.Excel, f_nom=50)
grid_model_2 = GridModel(grid_data_2) 

grid_data_3 = GridDataClass("tests/test_grid_3.xlsx", filetype=FileType.Excel, f_nom=50)
grid_model_3 = GridModel(grid_data_3) 

grid_data_4 = GridDataClass("tests/test_grid_4.xlsx", filetype=FileType.Excel, f_nom=50)
grid_model_4 = GridModel(grid_data_4) 

class TestDataClass(unittest.TestCase): 
    def test_data_class_model_1(self): 
        self.assertEqual(grid_data_1.get_PQ_mask(), (np.array([1]), np.array([1])))
        P, Q = grid_data_1.get_PQ_vals()
        self.assertTrue((P == np.array([-0. , -0.5])).all())
        self.assertTrue((Q == np.array([-0. , -0.5])).all())
        V_m, d_m, N = grid_data_1.get_V_delta_mask()
        self.assertEqual(N, 1)
        self.assertTrue((V_m == np.array([1])).all())
        self.assertTrue((d_m == np.array([1])).all())
        V, d = grid_data_1.get_V_delta_vals()
        self.assertTrue((V == np.array([1., 1.])).all())
        self.assertTrue((d == np.array([0., 0.])).all())

    def test_y_bus_model_1(self): 
        y_bus = grid_data_1.get_Y_bus() 
        y_bus_res = np.array([[ 1.4235294-5.6941175j, -1.4235294+5.6941175j],
                              [-1.4235294+5.6941175j,  1.4235294-5.6941175j]], dtype=np.complex64)
        self.assertTrue((np.abs(y_bus - y_bus_res) < 1e-6).all()) # Checks if all elements are less than 1e-6

    def test_y_bus_model_2(self): 
        y_bus = grid_data_2.get_Y_bus() 
        y_bus_res = np.array([[ 1.4235294 -5.6941175j, -0.7117647 +2.8470588j, -0.7117647 +2.8470588j],
                              [-0.7117647 +2.8470588j,  1.067647  -4.270588j , -0.35588235+1.4235294j],
                              [-0.7117647 +2.8470588j, -0.35588235+1.4235294j, 1.067647  -4.270588j ]], dtype=np.complex64)
        self.assertTrue((np.abs(y_bus - y_bus_res) < 1e-6).all()) # Checks if all elements are less than 1e-6


class TestGridModel(unittest.TestCase): 
    def test_basic_model1(self): 
        pf_res = grid_model_1.powerflow() 
        self.assertTrue((np.abs((pf_res.P_calc - np.array([0.52667153, -0.5])*pf_res.S_base)) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.Q_calc - np.array([0.60668611, -0.5])*pf_res.S_base)) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.V_buses - np.array([1., 0.8801433])))   < 1e-6).all())
        self.assertTrue((np.abs((pf_res.d_buses - np.array([0., -0.07048264]))) < 1e-6).all())

    def test_basic_model2(self): 
        pf_res = grid_model_2.powerflow() 
        self.assertTrue((np.abs((pf_res.P_calc -  np.array([0.43693196, 10., -10.]))) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.Q_calc -  np.array([0.90532378,  10.84240404, -10.]))) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.V_buses - np.array([1.        , 1.05      , 0.94256155])))   < 1e-6).all())
        self.assertTrue((np.abs((pf_res.d_buses - np.array([0.        ,  0.02831531, -0.0333914]))) < 1e-6).all())

    def test_basic_model3(self): 
        pf_res = grid_model_3.powerflow() 
        self.assertTrue((np.abs((pf_res.P_calc -  np.array([10.39785418, -50., 40., 5., -5.]))) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.Q_calc -  np.array([0.36225115, -20., 4.67633458, 18.69198958, -2.]))) < 1e-6).all())
        self.assertTrue((np.abs((pf_res.V_buses - np.array([1. , 0.99667064, 1., 1.02, 0.97060469])))   < 1e-6).all())
        self.assertTrue((np.abs((pf_res.d_buses - np.pi/180*np.array([0., -0.68015495, -0.16768721, -0.92423472, -2.59850105]))) < 1e-6).all())


if __name__ == "__main__": 
    unittest.main() 