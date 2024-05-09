from scipy import optimize
from types import SimpleNamespace
import numpy as np
import random

class ISLMClass:
    def __init__(self):
        self.par = SimpleNamespace()  # parameters
        self.sim = SimpleNamespace()  # simulation variables
        self.datamoms = SimpleNamespace()  # moments in the data
        self.moms = SimpleNamespace()  # moments in the model

    def ISLM_equations(self, variables, Y, C, I, G, r, T, P, M):
        """
        Args:
            variables (list or tuple): Contains two variables to be solved for: Output and Interest Rate
            Y (float): Output
            C (float): Consumption
            I (float): Investments
            G (float): Government spending
            r (float): Interest rate
            T (float): Taxes
            P (float): Consumer price index
            M (float): Money supply
        
        Returns:
            ISLM for r_IS = r_LM
        """
        Y, r = variables  # Unpack variables assumed to be Output (Y) and Interest Rate (r)
        r_IS = C + I + G - T  # Simplified IS equation: Y = C + I + G - T
        r_LM = M / (P * Y) - r  # Simplified LM equation: M = P * Y * r
        
        return [Y - r_IS, r_LM]  # Return residuals for both equations

    def multi_start(self, num_guesses=100, bounds=[(0, 1000), (0, 10)], fun=None):
        """
        Performs multi-start optimization to find the equilibrium solutions for Output and Interest Rate.
        
        Args:
            num_guesses (int): The number of random initial guesses
            bounds (list of tuple): The bounds for the random initial guesses for Y and r
            fun (function): The function to be optimized, default uses ISLM_equations
        
        Returns:
            Optimal Output and Interest Rate values
        """
        if fun is None:
            fun = self.ISLM_equations

        smallest_residual = np.inf
        best_solution = None

        # Loop through each random initial guess
        for _ in range(num_guesses):
            initial_guess = [np.random.uniform(low, high) for low, high in bounds]
            sol = optimize.root(fun=fun, x0=initial_guess, args=(self.sim.Y, self.sim.C, self.sim.I, self.sim.G, self.sim.r, self.sim.T, self.sim.P, self.sim.M), method='hybr')
            residual_norm = np.linalg.norm(sol.fun)
            if residual_norm < smallest_residual:
                smallest_residual = residual_norm
                best_solution = sol.x

        return best_solution, smallest_residual