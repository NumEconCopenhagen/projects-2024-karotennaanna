from scipy import optimize
from types import SimpleNamespace

import numpy as np

class ISLMClass:

    def __init__(self):
        
        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

    def ISLM_equations(variables, Y, C, I, G, r, T, P, M):
    """
    Args:
        variables (list or tuple): Contains two variables to be solved for: physical capital and human capital
        Y               (float): Output 
        C               (float): Consumption 
        I               (float): Investments 
        G                 (float): Goverment spending
        r                 (float): Interest rate
        T             (float): Taxes
        P             (float): Consumer price index 
        M            (float): Money supply 
    
    Returns:
        ISLM for r_IS=r_LM
    """
    # Variables to be solved for: Output and interest rate
    Y, r = variables
    
    # Checks for edge cases, used in multi_start
    if Y <= 0 or r <= 0:
        # Return a very large residual to indicate a poor solution
        return [np.inf, np.inf]

    # Set Solow equations for k_{t+1}-k_{t} = 0 and h_{t+1}-h_{t} = 0 
    r_IS=1/f*(e*Y-(M/P))
    r_LM=1/d**(a+c+G-b*T)-(1-b)*Y)

    # Return equations
    return r_IS, r_LM



    return result

