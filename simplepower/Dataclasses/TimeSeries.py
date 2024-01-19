import pandas as pd 
import numpy as np 
from typing import Sequence, Union

class PQTimeSeries: 
    def __init__(self, p_vals_mw: Union[float, np.ndarray], 
                 q_vals_mvar: Union[float, np.ndarray]): 
        if type(p_vals_mw) is float and type(q_vals_mvar) is float:   
            self.p_vals_mw = p_vals_mw 
            self.q_vals_mvar = q_vals_mvar
            self._len = -1 

        # Forces both p and q to be of same length if one value is float  
        elif type(p_vals_mw) is float: 
            self._len = len(q_vals_mvar)
            self.p_vals_mw = np.array([p_vals_mw for _ in range(self._len)])
            self.q_vals_mvar = q_vals_mvar

        elif type(q_vals_mvar) is float: 
            self._len = len(p_vals_mw)
            self.p_vals_mw = p_vals_mw
            self.q_vals_mvar = np.array([q_vals_mvar for _ in range(self._len)])

        else: 
            self.p_vals_mw = p_vals_mw
            self.q_vals_mvar = q_vals_mvar
            self._len = len(self.p_vals_mw)

        self._index = 0 

    def len(self):
        return self._len 
    
    def iloc(self, idx): 
        if idx < self.len():
            return (self.p_vals_mw[idx], self.q_vals_mvar[idx])
        else: 
            return (None, None)
    

if __name__ == "__main__": 
    p_data = np.array([0, 1, 2])
    q_data = np.array([3, 4, 5])

    ts = PQTimeSeries(p_data, q_data) 
    print(ts.iloc(3))

