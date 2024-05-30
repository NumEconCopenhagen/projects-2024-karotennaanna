import sympy as sp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

class ISLM_alg:
    
    par = SimpleNamespace()
    
    def __init__(self, a, b, c, d, e, f, T, G, M, P):
        """
        Initialize the ISLM_alg class with specified parameters.

        Parameters:
        a, b, c, d, e, f: Parameters for consumption, investment, and money demand functions.
        T: Lump-sum tax.
        G: Government spending.
        M: Money supply.
        P: Price level.
        """
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
        """
        Derive the IS curve equation by substituting consumption and investment into the goods market equilibrium.
        
        Returns:
        EqGoods_final: The final IS curve equation.
        """
        EqGoods_sub = self.EqGoods.subs(self.PE, self.C + self.I + self.G)
        EqGoods_subsub = EqGoods_sub.subs(self.C, self.a + self.b * (self.Y - self.T))
        EqGoods_final = EqGoods_subsub.subs(self.I, self.c - self.d * self.r)
        return EqGoods_final

    def derive_LM(self):
        """
        Derive the LM curve equation by substituting money supply and money demand into the money market equilibrium.
        
        Returns:
        dMoney_substituted: The final LM curve equation.
        """
        dMoney_substituted = self.dMoney.subs(self.L, self.M / self.P)
        return dMoney_substituted

    def solve_for_Y(self):
        """
        Solve for output (Y) from the IS and LM equations.
        
        Returns:
        IS_solution_Y, LM_solution_Y: Solutions for output (Y) from the IS and LM equations.
        """
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_Y = sp.solve(IS_eq, self.Y)
        LM_solution_Y = sp.solve(LM_eq, self.Y)
        return IS_solution_Y, LM_solution_Y

    def solve_for_r(self):
        """
        Solve for the interest rate (r) from the IS and LM equations.
        
        Returns:
        IS_solution_r, LM_solution_r: Solutions for the interest rate (r) from the IS and LM equations.
        """
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_r = sp.solve(IS_eq, self.r)
        LM_solution_r = sp.solve(LM_eq, self.r)
        return IS_solution_r, LM_solution_r

    def find_equilibrium(self):
        """
        Find the equilibrium values for output (Y) and the interest rate (r) by solving the IS and LM equations simultaneously.
        
        Returns:
        equilibrium: A list of dictionaries containing the equilibrium values for Y and r.
        """
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        equilibrium = sp.solve([IS_eq, LM_eq], (self.Y, self.r), dict=True)
        return equilibrium

    def objective(self, params):
        """
        Objective function to minimize, used for optimizing the lump-sum tax (T) to achieve a target interest rate.
        
        Parameters:
        params: List of parameters to optimize (in this case, T).
        
        Returns:
        The squared difference between the current interest rate and the target interest rate (0.04).
        """
        self.T = params[0]
        equilibrium = self.find_equilibrium()
        if equilibrium:
            for sol in equilibrium:
                if self.r in sol:
                    return (sol[self.r] - 0.04)**2
        return float('inf')

    def optimize_parameters(self):
        """
        Optimize the parameter T to achieve a target interest rate (0.04) using the Nelder-Mead method.
        
        Returns:
        result.x: The optimized value of T.
        """
        initial_guess = [self.T]
        result = minimize(self.objective, initial_guess, method='Nelder-Mead')
        self.T = result.x[0]
        return result.x

    def store_curves(self, x_range=(0.5, 2.5), num_points=100):
        """
        Store the IS and LM curve values over a specified range of output (Y).
        
        Parameters:
        x_range: Range of output (Y) values to consider.
        num_points: Number of points to compute within the range.
        
        Returns:
        IS_r_values, LM_r_values: Lists of interest rate (r) values corresponding to the IS and LM curves.
        """
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
        """
        Plot the IS and LM curves and mark the equilibrium points.
        
        Parameters:
        x_range: Range of output (Y) values to plot.
        num_points: Number of points to plot within the range.
        label_suffix: Suffix to add to the curve labels.
        IS_r_values: Interest rate (r) values for the IS curve.
        LM_r_values: Interest rate (r) values for the LM curve.
        equilibrium: List of dictionaries containing the equilibrium values for Y and r.
        y_range: Range of interest rate (r) values to plot.
        """
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
        plt.title('Baseline IS-LM Model')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits

        # Show the plot
        plt.show()

    def compare_G_changes(self, initial_G, new_G, x_range=(0.5, 2.5), num_points=100, y_range=(-2, 2)):
        """
        Compare the IS and LM curves before and after a fiscal shock (increase in government spending, G).
        
        Parameters:
        initial_G: Initial value of government spending.
        new_G: New value of government spending after the fiscal shock.
        x_range: Range of output (Y) values to plot.
        num_points: Number of points to plot within the range.
        y_range: Range of interest rate (r) values to plot.
        """
        print(f"IS-LM Model with fiscal shock (increase in G)")
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
        plt.title('IS-LM Model with fiscal shock (increase in G)')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits

        # Show the plot
        plt.show()

    def compare_M_changes(self, initial_M, new_M, x_range=(0.5, 2.5), num_points=100, y_range=(-2, 2)):
        """
        Compare the IS and LM curves before and after a monetary shock (increase in money supply, M).
        
        Parameters:
        initial_M: Initial value of money supply.
        new_M: New value of money supply after the monetary shock.
        x_range: Range of output (Y) values to plot.
        num_points: Number of points to plot within the range.
        y_range: Range of interest rate (r) values to plot.
        """
        print(f"IS-LM Model with fiscal shock (increase in M)")
        
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
        plt.title('IS-LM Model with fiscal shock (increase in M)')
        plt.legend()
        plt.grid(True)
        plt.xlim(x_range)  # Set x-axis limits
        plt.ylim(y_range)  # Set y-axis limits
        
        # Show the plot
        plt.show()

    def initialize_parameters(self, a, b, c, d, f, g, epsilon, e, M, P, L, T, Y, r, PE, C, I, G, NX):
        """
        Initialize additional parameters for the open economy.

        Parameters:
        a, b, c, d, f, g, epsilon, e: Parameters for consumption, investment, net exports, and money demand functions.
        M: Money supply.
        P: Price level.
        L: Symbol for money demand.
        T: Lump-sum tax.
        Y: Symbol for output.
        r: Symbol for interest rate.
        PE: Symbol for planned expenditure.
        C: Symbol for consumption.
        I: Symbol for investment.
        G: Government spending.
        NX: Symbol for net exports.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.f = f
        self.g = g
        self.epsilon = epsilon
        self.e = e
        self.M = M
        self.P = P
        self.L = L
        self.T = T
        self.Y = Y
        self.r = r
        self.PE = PE
        self.C = C
        self.I = I
        self.G = G
        self.NX = NX

        # Define equations for open economy
        self.PlannedExp_open = sp.Eq(self.PE, self.C + self.I + self.G + self.NX)
        self.PrivateCons_open = sp.Eq(self.C, self.a + self.b * (self.Y - self.T))
        self.Netexport = sp.Eq(self.NX, self.f - self.g * self.epsilon)
        
        # Define the money market equations
        self.EqMoney = sp.Eq(self.M / self.P, self.L)
        self.dMoney = sp.Eq(self.L, self.e * self.Y - self.f * self.r)

    def derive_IS_open(self):
        """
        Derive the IS curve equation for the open economy by substituting consumption, investment, and net exports into the goods market equilibrium.
        
        Returns:
        EqGoods_open_final: The final IS curve equation for the open economy.
        """
        EqGoods_open = sp.Eq(self.Y, self.PE)
        EqGoods_open_sub = EqGoods_open.subs(self.PE, self.C + self.I + self.G + self.NX)
        EqGoods_open_subsub = EqGoods_open_sub.subs(self.C, self.a + self.b * (self.Y - self.T))
        EqGoods_open_final = EqGoods_open_subsub.subs(self.I, self.c - self.d * self.r)
        EqGoods_open_final = EqGoods_open_final.subs(self.NX, self.f - self.g * self.epsilon)
        return EqGoods_open_final

    def derive_LM_open(self):
        """
        Derive the LM curve equation for the open economy by substituting money supply and money demand into the money market equilibrium.
        
        Returns:
        dMoney_substituted: The final LM curve equation for the open economy.
        """
        dMoney_substituted = self.dMoney.subs(self.L, self.M / self.P)
        return dMoney_substituted

    def solve_for_Y_open(self):
        """
        Solve for output (Y) from the IS and LM equations for the open economy.
        
        Returns:
        IS_solution_Y_open, LM_solution_Y_open: Solutions for output (Y) from the IS and LM equations for the open economy.
        """
        IS_eq_open = self.derive_IS_open()
        LM_eq_open = self.derive_LM_open()
        IS_solution_Y_open = sp.solve(IS_eq_open, self.Y)
        LM_solution_Y_open = sp.solve(LM_eq_open, self.Y)
        return IS_solution_Y_open, LM_solution_Y_open

    def solve_for_r_open(self):
        """
        Solve for the interest rate (r) from the IS and LM equations for the open economy.
        
        Returns:
        IS_solution_r_open, LM_solution_r_open: Solutions for the interest rate (r) from the IS and LM equations for the open economy.
        """
        IS_eq_open = self.derive_IS_open()
        LM_eq_open = self.derive_LM_open()
        IS_solution_r_open = sp.solve(IS_eq_open, self.r)
        LM_solution_r_open = sp.solve(LM_eq_open, self.r)
        return IS_solution_r_open, LM_solution_r_open

    def find_equilibrium_open(self):
        """
        Find the equilibrium values for output (Y) and the interest rate (r) in the open economy by solving the IS and LM equations simultaneously.
        
        Returns:
        equilibrium_open: A list of dictionaries containing the equilibrium values for Y and r in the open economy.
        """
        IS_eq_open = self.derive_IS_open()
        LM_eq_open = self.derive_LM_open()
        equilibrium_open = sp.solve([IS_eq_open, LM_eq_open], (self.Y, self.r), dict=True)
        return equilibrium_open

    def objective_open(self, params):
        """
        Objective function to minimize, used for optimizing the lump-sum tax (T) to achieve a target interest rate in the open economy.
        
        Parameters:
        params: List of parameters to optimize (in this case, T).
        
        Returns:
        The squared difference between the current interest rate and the target interest rate (0.04) in the open economy.
        """
        self.T = params[0]
        equilibrium_open = self.find_equilibrium_open()
        if equilibrium_open:
            for sol in equilibrium_open:
                if self.r in sol:
                    return (sol[self.r] - 0.04)**2
        return float('inf')

    def optimize_parameters_open(self):
        """
        Optimize the parameter T to achieve a target interest rate (0.04) in the open economy using the Nelder-Mead method.
        
        Returns:
        result.x: The optimized value of T in the open economy.
        """
        initial_guess = [self.T]
        result = minimize(self.objective_open, initial_guess, method='Nelder-Mead')
        self.T = result.x[0]
        return result.x

    def store_curves_open(self, x_range=(0.5, 2.5), num_points=100):
        """
        Store the IS and LM curve values over a specified range of output (Y) for the open economy.
        
        Parameters:
        x_range: Range of output (Y) values to consider.
        num_points: Number of points to compute within the range.
        
        Returns:
        IS_r_values_open, LM_r_values_open: Lists of interest rate (r) values corresponding to the IS and LM curves in the open economy.
        """
        self.Y_values = np.linspace(x_range[0], x_range[1], num_points)
        IS_r_values_open = [
            sp.solve(self.derive_IS_open().subs(self.Y, Y_val), self.r)[0].evalf()
            for Y_val in self.Y_values
        ]
        LM_r_values_open = [
            sp.solve(self.derive_LM_open().subs(self.Y, Y_val), self.r)[0].evalf()
            for Y_val in self.Y_values
        ]
        return IS_r_values_open, LM_r_values_open

