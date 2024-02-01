import pandas as pd 
import numpy as np 
from typing import Union
from abc import ABC

class BaseTimeSeries(ABC): 
    def __init__(self, 
                 ts1: Union[float, np.ndarray, list, tuple], 
                 ts2: Union[float, np.ndarray, list, tuple]):
        """Contains two datasets, either PQ data or PV data. """ 
        self._ts1_arr = isinstance(ts1, (np.ndarray, list, tuple))
        self._ts2_arr = isinstance(ts2, (np.ndarray, list, tuple))

        self.ts1 = ts1
        self.ts2 = ts2
    
    def iloc(self, idx) -> (float, float): 
        ts1 = self.ts1[idx] if self._ts1_arr else self.ts1 
        ts2 = self.ts2[idx] if self._ts2_arr else self.ts2 
        return (ts1, ts2)


class PQTimeSeries(BaseTimeSeries): 
    def __init__(self, p_vals_mw: Union[float, np.ndarray], 
                 q_vals_mvar: Union[float, np.ndarray]): 
        super().__init__(p_vals_mw, q_vals_mvar)
    

class PVTimeSeries(BaseTimeSeries): 
    def __init__(self, p_vals_mw: Union[float, np.ndarray], 
                 v_vals_pu: Union[float, np.ndarray]): 
        super().__init__(p_vals_mw, v_vals_pu)
    

if __name__ == "__main__": 
    p_data = np.array([0, 1, 2])
    q_data = np.array([3, 4, 5])

    ts = PQTimeSeries(p_data, q_data) 
    print(ts.iloc(3))

