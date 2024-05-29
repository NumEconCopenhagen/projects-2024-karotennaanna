import sympy as sp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class ISLM_alg:
    
    par = SimpleNamespace()
    
    def __init__(self, a, b, c, d, e, f, T, G, M, P):
        # Assign parameters to instance variables
        self.a, self.b, self.c, self.d, self.e, self.f, self.T, self.G, self.M, self.P = a, b, c, d, e, f, T, G, M, P
        
        # Define symbols
        self.Y, self.r = sp.symbols('Y r')
        self.C, self.I, self.PE, self.L = sp.symbols('C I PE L')
        
        # Define equations
        self.PlannedExp = sp.Eq(self.PE, self.C + self.I + self.G)
        self.PrivateCons = sp.Eq(self.C, self.a + self.b * (self.Y - self.T))
        self.EqGoods = sp.Eq(self.Y, self.PE)
        self.Investment = sp.Eq(self.I, self.c - self.d * self.r)
        
        # Define the money market equations
        self.EqMoney = sp.Eq(self.M / self.P, self.L)
        self.dMoney = sp.Eq(self.L, self.e * self.Y - self.f * self.r)

    def derive_IS(self):
        # Substitute PE from PlannedExp into EqGoods
        EqGoods_sub = self.EqGoods.subs(self.PE, self.C + self.I + self.G)
        
        # Substitute the expression for C from PrivateCons into EqGoods_sub
        EqGoods_subsub = EqGoods_sub.subs(self.C, self.a + self.b * (self.Y - self.T))
        
        # Substitute I from Investment into EqGoods_subsub:
        EqGoods_final = EqGoods_subsub.subs(self.I, self.c - self.d * self.r)
        
        return EqGoods_final

    def derive_LM(self):
        # Substitute EqMoney into dMoney
        dMoney_substituted = self.dMoney.subs(self.L, self.M / self.P)
        
        return dMoney_substituted
    
    def solve_for_Y(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_Y = sp.solve(IS_eq, self.Y)
        LM_solution_Y = sp.solve(LM_eq, self.Y)
        return IS_solution_Y, LM_solution_Y

    def solve_for_r(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_r = sp.solve(IS_eq, self.r)
        LM_solution_r = sp.solve(LM_eq, self.r)
        return IS_solution_r, LM_solution_r

    def find_equilibrium(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        
        # Solve the IS and LM equations together for Y and r
        equilibrium = sp.solve([IS_eq, LM_eq], (self.Y, self.r), dict=True)
        
        return equilibrium
    
    def objective(self, params):
        self.T = params[0]
        
        equilibrium = self.find_equilibrium()
        if equilibrium:
            for sol in equilibrium:
                if self.r in sol:
                    return (sol[self.r] - 0.04)**2
        return float('inf')

    def optimize_parameters(self):
        initial_guess = [self.T]
        result = minimize(self.objective, initial_guess, method='Nelder-Mead')
        self.T = result.x[0]
        return result.x

    def store_curves(self, x_range=(0.5, 2.5), num_points=100):
        self.Y_values = np.linspace(x_range[0], x_range[1], num_points)
        IS_r_values = [
            sp.solve(self.derive_IS().subs(self.Y, Y_val), self.r)[0].evalf()
            for Y_val in self.Y_values
        ]
        LM_r_values = [
            sp.solve(self.derive_LM().subs(self.Y, Y_val), self.r)[0].evalf()
            for Y_val in self.Y_values
        ]
        return IS_r_values, LM_r_values

    def plot(self, x_range=(0.5, 2.5), num_points=100, label_suffix='', IS_r_values=None, LM_r_values=None, equilibrium=None, y_range=(-2, 2)):
        plt.figure(figsize=(10, 6))

        # Plot IS and LM curves if provided
        if IS_r_values is not None and LM_r_values is not None:
            plt.plot(self.Y_values, IS_r_values, label=f'IS Curve {label_suffix}', color='blue')
            plt.plot(self.Y_values, LM_r_values, label=f'LM Curve {label_suffix}', color='red')

        # Plot equilibrium point if provided
        if equilibrium is not None:
            for sol in equilibrium:
                if self.r in sol and self.Y in sol:
                    plt.scatter(sol[self.Y], sol[self.r], color='green', s=100, 
                                label=f'Equilibrium {label_suffix} (Y={sol[self.Y]:.2f}, r={sol[self.r]:.2f})')

        # Add labels and legend
        plt.xlabel('Output (GDP)')
        plt.ylabel('Interest Rate, r')
        plt.title('IS-LM Model')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits

        # Show the plot
        plt.show()

    def compare_G_changes(self, initial_G, new_G, x_range=(0.5, 2.5), num_points=100, y_range=(-2, 2)):
        print(f"Comparing IS curves for G={initial_G} and G={new_G}...")
        self.G = initial_G
        initial_IS_r_values, initial_LM_r_values = self.store_curves(x_range, num_points)
        initial_equilibrium = self.find_equilibrium()

        self.G = new_G
        new_IS_r_values, new_LM_r_values = self.store_curves(x_range, num_points)
        new_equilibrium = self.find_equilibrium()

        plt.figure(figsize=(10, 6))
        plt.plot(self.Y_values, initial_IS_r_values, label=f'IS Curve G={initial_G}', color='blue', linestyle='--')
        plt.plot(self.Y_values, initial_LM_r_values, label=f'LM Curve G={initial_G}', color='red', linestyle='--')
        plt.plot(self.Y_values, new_IS_r_values, label=f'IS Curve G={new_G}', color='blue')
        plt.plot(self.Y_values, new_LM_r_values, label=f'LM Curve G={new_G}', color='red')

        # Mark the equilibrium points
        if initial_equilibrium:
            for sol in initial_equilibrium:
                if self.r in sol and self.Y in sol:
                    plt.scatter(sol[self.Y], sol[self.r], color='green', s=100, 
                                label=f'Equilibrium G={initial_G} (Y={sol[self.Y]:.2f}, r={sol[self.r]:.2f})')

        if new_equilibrium:
            for sol in new_equilibrium:
                if self.r in sol and self.Y in sol:
                    plt.scatter(sol[self.Y], sol[self.r], color='purple', s=100, 
                                label=f'Equilibrium G={new_G} (Y={sol[self.Y]:.2f}, r={sol[self.r]:.2f})')

        # Add labels and legend
        plt.xlabel('Output (GDP)')
        plt.ylabel('Interest Rate, r')
        plt.title('IS-LM Model Comparison for Different G Values')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits

        # Show the plot
        plt.show()

    def compare_M_changes(self, initial_M, new_M, x_range=(0.5, 2.5), num_points=100, y_range=(-2, 2)):
        print(f"Comparing IS-LM models for M={initial_M} and M={new_M} with G=1...")
        
        # Store initial conditions with G=1
        self.G = 1
        self.M = initial_M
        initial_IS_r_values, initial_LM_r_values = self.store_curves(x_range, num_points)
        initial_equilibrium = self.find_equilibrium()
        
        # Change M and find new equilibrium
        self.M = new_M
        new_IS_r_values, new_LM_r_values = self.store_curves(x_range, num_points)
        new_equilibrium = self.find_equilibrium()
        
        # Plot the curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.Y_values, initial_IS_r_values, label=f'IS Curve M={initial_M}', color='blue', linestyle='--')
        plt.plot(self.Y_values, initial_LM_r_values, label=f'LM Curve M={initial_M}', color='red', linestyle='--')
        plt.plot(self.Y_values, new_IS_r_values, label=f'IS Curve M={new_M}', color='blue')
        plt.plot(self.Y_values, new_LM_r_values, label=f'LM Curve M={new_M}', color='red')
        
        # Mark the equilibrium points
        if initial_equilibrium:
            for sol in initial_equilibrium:
                if self.r in sol and self.Y in sol:
                    plt.scatter(sol[self.Y], sol[self.r], color='green', s=100, 
                                label=f'Equilibrium M={initial_M} (Y={sol[self.Y]:.2f}, r={sol[self.r]:.2f})')
        
        if new_equilibrium:
            for sol in new_equilibrium:
                if self.r in sol and self.Y in sol:
                    plt.scatter(sol[self.Y], sol[self.r], color='purple', s=100, 
                                label=f'Equilibrium M={new_M} (Y={sol[self.Y]:.2f}, r={sol[self.r]:.2f})')
        
        # Add labels and legend
        plt.xlabel('Output (GDP)')
        plt.ylabel('Interest Rate, r')
        plt.title('IS-LM Model Comparison for Different M Values')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits
        
        # Show the plot
        plt.show()


