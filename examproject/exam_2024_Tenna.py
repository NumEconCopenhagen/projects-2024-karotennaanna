import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
from types import SimpleNamespace

class ProductionEconomyModel:
    def __init__(self, A=1.0, gamma=0.5, alpha=0.3, nu=1.0, epsilon=2.0, tau=0.0, T=0.0, w=1.0):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = w

    def labor_demand(self, p1, p2):
        """Calculate total labor demand for the economy based on prices and tax."""
        labor_demand = ((self.A * (p1**self.alpha) * ((p2 + self.tau)**(1 - self.alpha)) / self.nu)**(1 / (self.epsilon + 1)))
        return labor_demand

    def production(self, labor):
        """Calculate production of goods based on labor."""
        return self.A * (labor**self.gamma)

    def consumption(self, labor, p1, p2):
        """Calculate consumer's consumption of goods based on labor and prices."""
        income = self.w * labor + self.T
        c1 = self.alpha * income / p1
        c2 = (1 - self.alpha) * income / (p2 + self.tau)
        return c1, c2

    def check_market_clearing(self, p1, p2):
        """Check market clearing conditions for labor and goods markets."""
        labor_demand = self.labor_demand(p1, p2)
        production1 = self.production(labor_demand)
        production2 = self.production(labor_demand)
        consumption1, consumption2 = self.consumption(labor_demand, p1, p2)
        
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
            consumption1, consumption2 = self.consumption(labor_demand, p1, p2)
            excess_demand1 = production1 - consumption1
            excess_demand2 = production2 - consumption2
            return excess_demand1**2 + excess_demand2**2

        result = minimize(market_excess_demand, [1.0, 1.0], bounds=[(0.1, 2.0), (0.1, 2.0)])
        if result.success:
            return result.x
        else:
            raise ValueError("Equilibrium prices could not be found")

    def set_policy(self, tau, T):
        """Set the values for tax and transfer."""
        self.tau = tau
        self.T = T

    def evaluate_policy_impact(self, tau, T):
        """Evaluate the impact of a given tax and transfer policy on the equilibrium."""
        self.set_policy(tau, T)
        equilibrium_prices = self.solve_equilibrium()
        p1, p2 = equilibrium_prices
        labor_demand = self.labor_demand(p1, p2)
        production1 = self.production(labor_demand)
        production2 = self.production(labor_demand)
        consumption1, consumption2 = self.consumption(labor_demand, p1, p2)
        
        return {
            'Equilibrium Prices': equilibrium_prices,
            'Labor Demand': labor_demand,
            'Production Good 1': production1,
            'Production Good 2': production2,
            'Consumption Good 1': consumption1,
            'Consumption Good 2': consumption2
        }

    def social_welfare(self, labor, consumption1, consumption2):
        """Define the social welfare function."""
        utility = (consumption1**(1 - self.alpha)) * (consumption2**self.alpha) - (1 / (1 + self.epsilon)) * (labor**(1 + self.epsilon))
        return utility

    def maximize_social_welfare(self):
        """Find the values of tau and T that maximize the social welfare function."""

        def objective(policy_params):
            tau, T = policy_params
            self.set_policy(tau, T)
            equilibrium_prices = self.solve_equilibrium()
            labor_demand = self.labor_demand(*equilibrium_prices)
            consumption1, consumption2 = self.consumption(labor_demand, *equilibrium_prices)
            welfare = self.social_welfare(labor_demand, consumption1, consumption2)
            return -welfare  # Minimize the negative of social welfare

        result = minimize(objective, [self.tau, self.T], bounds=[(0, 1), (0, 1)])
        if result.success:
            optimal_tau, optimal_T = result.x
            return optimal_tau, optimal_T
        else:
            raise ValueError("Optimal policy could not be found")

# Example of how to use the class and its methods
if __name__ == "__main__":
    model = ProductionEconomyModel()
    equilibrium_prices = model.solve_equilibrium()
    print(f"Equilibrium prices: p1 = {equilibrium_prices[0]}, p2 = {equilibrium_prices[1]}")

    # Evaluate the impact of a specific policy
    policy_impact = model.evaluate_policy_impact(tau=0.1, T=0.5)
    print(f"Policy Impact: {policy_impact}")

    # Find the optimal tau and T
    optimal_tau, optimal_T = model.maximize_social_welfare()
    print(f"Optimal tau: {optimal_tau}, Optimal T: {optimal_T}")


import numpy as np

class ProductionEconomyModel2:
    def __init__(self, alpha, gamma, epsilon, nu, A, p1, p2, w, tau, T):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nu = nu
        self.A = A
        self.p1 = p1
        self.p2 = p2
        self.w = w
        self.tau = tau
        self.T = T

    def optimal_behavior(self, l):
        c1 = self.c1(l)
        c2 = self.c2(l)
        utility = np.log(c1**self.alpha * c2**(1 - self.alpha)) - self.nu * (l**(1 + self.epsilon)) / (1 + self.epsilon)
        return utility

    def c1(self, l):
        return self.alpha * (self.w * l + self.T + self.pi1() + self.pi2()) / self.p1

    def c2(self, l):
        return (1 - self.alpha) * (self.w * l + self.T + self.pi1() + self.pi2()) / (self.p2 + self.tau)

    def pi1(self):
        return (1 - self.gamma) / self.gamma * self.w * ((self.p1 * self.A * self.gamma / self.w)**(1 / (1 - self.gamma)))

    def pi2(self):
        return (1 - self.gamma) / self.gamma * self.w * ((self.p2 * self.A * self.gamma / self.w)**(1 / (1 - self.gamma)))

    def labor_market(self):
        l1_star = (self.p1 * self.A * self.gamma / self.w)**(1 / (1 - self.gamma))
        l2_star = (self.p2 * self.A * self.gamma / self.w)**(1 / (1 - self.gamma))
        return l1_star + l2_star

    def good_market_1(self):
        return self.c1_star() - self.y1_star()

    def good_market_2(self):
        return self.c2_star() - self.y2_star()

    def y1_star(self):
        return self.A * (self.labor_market() / 2)**self.gamma

    def y2_star(self):
        return self.A * (self.labor_market() / 2)**self.gamma

    def c1_star(self):
        return self.c1(self.labor_market())

    def c2_star(self):
        return self.c2(self.labor_market())