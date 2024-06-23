import numpy as np
from scipy import optimize
from types import SimpleNamespace

class ProductionEconomy:
    def __init__(self, par=None):
        self.par = par = SimpleNamespace()
        
        # Firms parameters
        par.A = 1.0
        par.gamma = 0.5

        # Households parameters
        par.alpha = 0.3
        par.nu = 1.0
        par.epsilon = 2.0

        # Government parameters
        par.tau = 0.0
        par.T = 0.0

        # Question 3 specific parameter
        par.kappa = 0.1

        self.w = 1  # Assuming wage is 1 as numeraire
        self.p1 = None
        self.p2 = None

    # Calculate household utility given labor and prices
    def utility_(self, l, p1, p2):
        c1 = self.c1(l, p1, p2)
        c2 = self.c2(l, p1, p2)
        return np.log(c1**self.par.alpha * c2**(1 - self.par.alpha)) - self.par.nu * (l**(1 + self.par.epsilon)) / (1 + self.par.epsilon)

    # Calculate consumption of good 1
    def c1(self, l, p1, p2):
        profit1 = self.firm_profit1(p1)
        profit2 = self.firm_profit2(p2)
        return self.par.alpha * (self.w * l + self.par.T + profit1 + profit2) / p1

    # Calculate consumption of good 2
    def c2(self, l, p1, p2):
        profit1 = self.firm_profit1(p1)
        profit2 = self.firm_profit2(p2)
        return (1 - self.par.alpha) * (self.w * l + self.par.T + profit1 + profit2) / (p2 + self.par.tau)

    # Optimize labor to maximize utility for workers
    def optimize_l(self, p1, p2):
        obj = lambda l: -self.utility_(l, p1, p2)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')
        l_star = res.x
        c1_star = self.c1(l_star, p1, p2)
        c2_star = self.c2(l_star, p1, p2)
        return l_star, c1_star, c2_star

    # Calculate firms' labor demand given prices
    def firm_labor_demand(self, p1, p2):
        l1_star = (p1 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        l2_star = (p2 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        return l1_star, l2_star

    # Calculate profit for firm 1 given price
    def firm_profit1(self, p1):
        l1_star = (p1 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        y1_star = self.par.A * (l1_star)**self.par.gamma
        return (1 - self.par.gamma) * y1_star * p1

    # Calculate profit for firm 2 given price
    def firm_profit2(self, p2):
        l2_star = (p2 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        y2_star = self.par.A * (l2_star)**self.par.gamma
        return (1 - self.par.gamma) * y2_star * p2

    # Check market clearing conditions for labor and good 1
    def market_clearing_conditions(self, p):
        p1, p2 = p
        l1_star, l2_star = self.firm_labor_demand(p1, p2)
        l_star, c1_star, c2_star = self.optimize_l(p1, p2)
        labor_market = l1_star + l2_star - l_star

        good_market_1 = c1_star - self.par.A * (l1_star)**self.par.gamma
        
        return [labor_market, good_market_1]

# Instantiate the ProductionEconomy class
economy = ProductionEconomy()

# Initial guess for prices
initial_guess = [1, 1]

# Find equilibrium prices
equilibrium_prices = optimize.root(economy.market_clearing_conditions, initial_guess).x
p1_star, p2_star = equilibrium_prices

# Get equilibrium labor and consumption
l_star, c1_star, c2_star = economy.optimize_l(p1_star, p2_star)

# Output results
equilibrium_results = {
    "Equilibrium Price p1 (p1_star)": p1_star,
    "Equilibrium Price p2 (p2_star)": p2_star,
    "Equilibrium Labor (l_star)": l_star,
    "Equilibrium Consumption c1 (c1_star)": c1_star,
    "Equilibrium Consumption c2 (c2_star)": c2_star
}

# Print the results in a formatted way
for key, value in equilibrium_results.items():
    print(f"{key}: {value:.4f}")

