import numpy as np
from scipy.optimize import minimize

class EconomicModel:
    def __init__(self, alpha, nu, epsilon, A, gamma):
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.A = A
        self.gamma = gamma
    
    def optimal_labor(self, p1, p2, w, tau, T):
        # Calculate optimal labor
        return ((p1 * self.A * self.gamma) / w) ** (1 / self.gamma), ((p2 * self.A * self.gamma) / w) ** (1 / self.gamma)

    def optimal_profits(self, p, w):
        # Calculate optimal profits
        return (1 - self.gamma) / self.gamma * w * (p * self.A * self.gamma / w) ** (1 - self.gamma)

    def consumption(self, w, T, p1, p2, tau):
        # Calculate consumption
        labor1, labor2 = self.optimal_labor(p1, p2, w, tau, T)
        profits1 = self.optimal_profits(p1, w)
        profits2 = self.optimal_profits(p2, w)
        c1 = self.alpha * (w * (labor1 + labor2) + T + profits1 + profits2) / p1
        c2 = (1 - self.alpha) * (w * (labor1 + labor2) + T + profits1 + profits2) / (p2 + tau)
        return c1, c2
    
    def utility(self, c1, c2, l):
        # Calculate utility
        return self.alpha * np.log(c1) + (1 - self.alpha) * np.log(c2) - self.nu * l ** (1 + self.epsilon) / (1 + self.epsilon)
    
    def labor_demand(self, p1, p2):
        # Dummy labor demand calculation
        return 1.0  # Placeholder, you should replace this with your actual labor demand function

    def production(self, labor_demand):
        # Dummy production function
        return labor_demand  # Placeholder, you should replace this with your actual production function

    def check_market_clearing(self, p1, p2):
        """Check market clearing conditions for labor and goods markets."""
        labor_demand = self.labor_demand(p1, p2)
        production1 = self.production(labor_demand)
        production2 = self.production(labor_demand)
        consumption1, consumption2 = self.consumption(1.0, 0.5, p1, p2, 0.1)  # Placeholder values for w, T, tau
        
        return {
            'Labor Market': np.isclose(labor_demand, labor_demand),
            'Good Market 1': np.isclose(production1, consumption1),
            'Good Market 2': np.isclose(production2, consumption2)
        }

    def solve_equilibrium(self):
        """
        Solve for the equilibrium prices p1 and p2 that clear the markets.
        """

        def market_excess_demand(prices):
            p1, p2 = prices
            labor_demand = self.labor_demand(p1, p2)
            production1 = self.production(labor_demand)
            production2 = self.production(labor_demand)
            consumption1, consumption2 = self.consumption(1.0, 0.5, p1, p2, 0.1)  # Placeholder values for w, T, tau
            excess_demand1 = production1 - consumption1
            excess_demand2 = production2 - consumption2
            return excess_demand1**2 + excess_demand2**2

        result = minimize(market_excess_demand, [1.0, 1.0], bounds=[(0.1, 2.0), (0.1, 2.0)])
        if result.success:
            return result.x
        else:
            raise ValueError("Equilibrium prices could not be found")

# Example usage:
alpha = 0.5
nu = 0.1
epsilon = 1.5
A = 1.0
gamma = 0.3

model = EconomicModel(alpha, nu, epsilon, A, gamma)
p1, p2 = model.solve_equilibrium()
print(f"p1={p1:.3f} and p2={p2:.3f}")
