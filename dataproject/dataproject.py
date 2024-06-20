def keep_regs(df, regs):
    """
    Filter a DataFrame to exclude rows containing specified regions.

    Args:
        df (pd.DataFrame): The input DataFrame containing a column 'reg' with region names.
        regs (list): A list of region substrings to exclude from the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame excluding rows where the 'reg' column contains any of the specified substrings.
    """
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df

from types import SimpleNamespace
import numpy as np
import pandas as pd

class Solowclass:
    """
    A class to handle Solow growth model data and parameters.

    Attributes:
        par (SimpleNamespace): A namespace for model parameters.
        sim (SimpleNamespace): A namespace for simulation variables.
        datamoms (SimpleNamespace): A namespace for moments in the data.
        moms (SimpleNamespace): A namespace for moments in the model.
        data (pd.DataFrame): A DataFrame holding the input data from a file.
    """

    def __init__(self,filename):
        """
        Initializes the Solowclass with data from a file.

        Args:
            filename (str): Path to the file (e.g., CSV) containing the data.

        Returns:
            None
        """
        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model
        
        # Reading the Excel file into a DataFrame and displaying the first 5 rows
        self.data = pd.read_csv(filename, delimiter=";")

        # a. externally given parameters
        par.alpha = None
    
    def print_head(self):
        """
        Prints the first 5 rows of the data.

        Args:
            None

        Returns:
            None
        """
        print(self.data.head(5))


import numpy as np

class AlphaCalculator:
    """
    A class to calculate the alpha value for economic models using input data.

    Attributes:
        data (pd.DataFrame): Input data containing necessary columns for calculations.
    """
    def __init__(self, data):
        """
        Initializes the AlphaCalculator with input data.

        Args:
            data (pd.DataFrame): A DataFrame containing columns for Q, B, K, and L.

        Returns:
            None
        """
        self.data = data
        
    def calculate_alpha(self, Q_col, B_col, K_col, L_col):
        """
        Calculates the alpha value based on provided columns for Q, B, K, and L.

        Args:
            Q_col (str): The name of the column representing output (Q).
            B_col (str): The name of the column representing productivity (B).
            K_col (str): The name of the column representing capital (K).
            L_col (str): The name of the column representing labor (L).

        Returns:
            float: The calculated alpha value.
        """ 
        ln_Q_t = np.log(self.data[Q_col])
        ln_B_t = np.log(self.data[B_col])
        ln_KL_t = np.log(self.data[K_col] * self.data[L_col])
        
        alpha_value = (ln_Q_t - ln_KL_t) / (ln_B_t - ln_KL_t)
        return alpha_value