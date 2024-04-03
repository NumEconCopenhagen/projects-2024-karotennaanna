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

class ASADClass:

    def __init__(self):

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

        # a. externally given parameters
        par.alpha = None

    def GDP(self):

    return error 