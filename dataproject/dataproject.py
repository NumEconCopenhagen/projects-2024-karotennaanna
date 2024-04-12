def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.
    
    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df

from types import SimpleNamespace
import numpy as np
import pandas as pd

class Solowclass:

    def __init__(self,filename):

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model
        
        # Reading the Excel file into a DataFrame and displaying the first 5 rows
        self.data = pd.read_csv(filename, delimiter=";")

        # a. externally given parameters
        par.alpha = None
    
    def print_head(self):
        print(self.data.head(5))


import numpy as np

class AlphaCalculator:
    def __init__(self, data):
        # Initialize the AlphaCalculator class with input data
        self.data = data
        
    def calculate_alpha(self, Q_col, B_col, K_col, L_col):
        # Calculate alpha value using the specified columns for Q, B, K, and L
        ln_Q_t = np.log(self.data[Q_col])
        ln_B_t = np.log(self.data[B_col])
        ln_KL_t = np.log(self.data[K_col] * self.data[L_col])
        
        alpha_value = (ln_Q_t - ln_KL_t) / (ln_B_t - ln_KL_t)
        return alpha_value

