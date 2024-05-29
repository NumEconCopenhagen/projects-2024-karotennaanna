import sympy as sp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class ISLM_alg:
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
    
class ISLMPlotter:
    def __init__(self, model):
        self.model = model
        self.initial_IS_r_values = None
        self.initial_LM_r_values = None

    def store_initial_curves(self, Y_range=(0, 3), num_points=100):
        Y_values = np.linspace(Y_range[0], Y_range[1], num_points)
        self.initial_IS_r_values = [sp.solve(self.model.derive_IS().subs(self.model.Y, Y_val), self.model.r)[0].evalf() for Y_val in Y_values]
        self.initial_LM_r_values = [sp.solve(self.model.derive_LM().subs(self.model.Y, Y_val), self.model.r)[0].evalf() for Y_val in Y_values]

    def plot(self, Y_range=(0, 3), num_points=100, label_suffix='', include_initial_curves=False):
        new_equilibrium = self.model.find_equilibrium()

        # Initialize new_Y and new_r
        new_Y, new_r = None, None

        # Display the new equilibrium solutions
        if new_equilibrium:
            for sol in new_equilibrium:
                if self.model.r in sol:
                    new_r = sol[self.model.r]
                if self.model.Y in sol:
                    new_Y = sol[self.model.Y]
        
        # Ensure new_Y and new_r are defined before plotting
        if new_Y is not None and new_r is not None:
            # Generate a range of Y values
            Y_values = np.linspace(Y_range[0], Y_range[1], num_points)

            # Calculate corresponding r values for IS and LM curves
            IS_r_values = [sp.solve(self.model.derive_IS().subs(self.model.Y, Y_val), self.model.r)[0].evalf() for Y_val in Y_values]
            LM_r_values = [sp.solve(self.model.derive_LM().subs(self.model.Y, Y_val), self.model.r)[0].evalf() for Y_val in Y_values]

            # Plot the curves
            plt.figure(figsize=(10, 6))
            if include_initial_curves and self.initial_IS_r_values and self.initial_LM_r_values:
                plt.plot(Y_values, self.initial_IS_r_values, label='Initial IS Curve', color='blue', linestyle='--')
                plt.plot(Y_values, self.initial_LM_r_values, label='Initial LM Curve', color='red', linestyle='--')
            plt.plot(Y_values, IS_r_values, label=f'IS Curve {label_suffix}', color='blue')
            plt.plot(Y_values, LM_r_values, label=f'LM Curve {label_suffix}', color='red')

            # Mark the new equilibrium point
            plt.scatter(new_Y, new_r, color='green', s=100, label=f'New Equilibrium {label_suffix} (Y={new_Y:.2f}, r={new_r:.2f})')

            # Add labels and legend
            plt.xlabel('Output, Y')
            plt.ylabel('Interest Rate, r')
            plt.title('IS-LM Model')
            plt.legend()
            plt.grid(True)

            # Show the plot
            plt.show()
        else:
            print("New equilibrium values for Y and r are not defined.")

    def compare_G_changes(self, initial_G, new_G, Y_range=(0, 3), num_points=100):
        self.model.G = initial_G
        self.store_initial_curves(Y_range, num_points)

        self.model.G = new_G
        self.plot(Y_range, num_points, label_suffix=f'G={new_G}', include_initial_curves=True)

    def compare_M_changes(self, initial_M, new_M, Y_range=(0, 3), num_points=100):
        self.model.M = initial_M
        self.store_initial_curves(Y_range, num_points)

        self.model.M = new_M
        self.plot(Y_range, num_points, label_suffix=f'M={new_M}', include_initial_curves=True)
class ISLM_open:
    def __init__(self, a, b, c, d, e, f, g, T, G, M, P, r, N, X, IM, e_rate, i_foreign):
        # Assign parameters to instance variables
        self.par = SimpleNamespace(a=a, b=b, c=c, d=d, e=e, f=f, g=g, T=T, G=G, M=M, P=P, r=r, N=N, X=X, IM=IM, e_rate=e_rate, i_foreign=i_foreign)
        
        # Define symbols
        self.Y, self.r = sp.symbols('Y r')
        self.C, self.I, self.PE, self.L = sp.symbols('C I PE L')
        
        # Define equations
        self.PlannedExp = sp.Eq(self.PE, self.C + self.I + self.G + self.NX)
        self.PrivateCons = sp.Eq(self.C, self.a + self.b * (self.Y - self.T))
        self.EqGoods = sp.Eq(self.Y, self.PE)
        self.Investment = sp.Eq(self.I, self.c - self.d * self.r)
        self.NetExports = sp.Eq(self.NX, self.X * self.e_rate - self.IM * self.Y)
        
        # Define the money market equations
        self.EqMoney = sp.Eq(self.M / self.P, self.L)
        self.dMoney = sp.Eq(self.L, self.e * self.Y - self.f * (self.r - self.i_foreign))

    def derive_IS(self):
        # Substitute PE and NX from PlannedExp and NetExports into EqGoods
        EqGoods_sub = self.EqGoods.subs(self.PE, self.C + self.I + self.G + self.NX)
        EqGoods_sub = EqGoods_sub.subs(self.NX, self.X * self.e_rate - self.IM * self.Y)
        
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

class ISLMPlotter_open:
    def __init__(self, model):
        self.model = model
        self.initial_IS_r_values = None
        self.initial_LM_r_values = None

    def store_initial_curves(self, Y_range=(0, 3), num_points=100):
        Y_values = np.linspace(Y_range[0], Y_range[1], num_points)
        self.initial_IS_r_values = [sp.solve(self.model.derive_IS().subs(self.model.Y, Y_val), self.model.par.r)[0].evalf() for Y_val in Y_values]
        self.initial_LM_r_values = [sp.solve(self.model.derive_LM().subs(self.model.Y, Y_val), self.model.par.r)[0].evalf() for Y_val in Y_values]

    def plot(self, Y_range=(0, 3), num_points=100, label_suffix='', include_initial_curves=False):
        new_equilibrium = self.model.find_equilibrium()

        # Initialize new_Y and new_r
        new_Y, new_r = None, None

        # Display the new equilibrium solutions
        if new_equilibrium:
            for sol in new_equilibrium:
                if self.model.par.r in sol:
                    new_r = sol[self.model.par.r]
                if self.model.Y in sol:
                    new_Y = sol[self.model.Y]
        
        # Ensure new_Y and new_r are defined before plotting
        if new_Y is not None and new_r is not None:
            # Generate a range of Y values
            Y_values = np.linspace(Y_range[0], Y_range[1], num_points)

            # Calculate corresponding r values for IS and LM curves
            IS_r_values = [sp.solve(self.model.derive_IS().subs(self.model.Y, Y_val), self.model.par.r)[0].evalf() for Y_val in Y_values]
            LM_r_values = [sp.solve(self.model.derive_LM().subs(self.model.Y, Y_val), self.model.par.r)[0].evalf() for Y_val in Y_values]

            # Plot the curves
            plt.figure(figsize=(10, 6))
            if include_initial_curves and self.initial_IS_r_values and self.initial_LM_r_values:
                plt.plot(Y_values, self.initial_IS_r_values, label='Initial IS Curve', color='blue', linestyle='--')
                plt.plot(Y_values, self.initial_LM_r_values, label='Initial LM Curve', color='red', linestyle='--')
            plt.plot(Y_values, IS_r_values, label=f'IS Curve {label_suffix}', color='blue')
            plt.plot(Y_values, LM_r_values, label=f'LM Curve {label_suffix}', color='red')

            # Mark the new equilibrium point
            plt.scatter(new_Y, new_r, color='green', s=100, label=f'New Equilibrium {label_suffix} (Y={new_Y:.2f}, r={new_r:.2f})')

            # Add labels and legend
            plt.xlabel('Output, Y')
            plt.ylabel('Interest Rate, r')
            plt.title('IS-LM Model')
            plt.legend()
            plt.grid(True)

            # Show the plot
            plt.show()
        else:
            print("New equilibrium values for Y and r are not defined.")

    def compare_G_changes(self, initial_G, new_G, Y_range=(0, 3), num_points=100):
        self.model.par.G = initial_G
        self.store_initial_curves(Y_range, num_points)

        self.model.par.G = new_G
        self.plot(Y_range, num_points, label_suffix=f'G={new_G}', include_initial_curves=True)

    def compare_M_changes(self, initial_M, new_M, Y_range=(0, 3), num_points=100):
        self.model.par.M = initial_M
        self.store_initial_curves(Y_range, num_points)

        self.model.par.M = new_M
        self.plot(Y_range, num_points, label_suffix=f'M={new_M}', include_initial_curves=True)    

