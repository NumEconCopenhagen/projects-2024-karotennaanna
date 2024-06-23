import numpy as np
from scipy import optimize
from types import SimpleNamespace

class ProductionEconomy:
    def __init__(self, par=None):
        """
        Initializes the ProductionEconomy class with default parameters.
        
        Parameters:
        par (SimpleNamespace, optional): Namespace to store parameters. Defaults to None.
        """
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

    def utility_(self, l, p1, p2):
        """
        Calculate household utility given labor and prices.

        Parameters:
        l (float): Labor supplied by the household.
        p1 (float): Price of good 1.
        p2 (float): Price of good 2.

        Returns:
        float: Utility value.
        """
        c1 = self.c1(l, p1, p2)
        c2 = self.c2(l, p1, p2)
        return np.log(c1**self.par.alpha * c2**(1 - self.par.alpha)) - self.par.nu * (l**(1 + self.par.epsilon)) / (1 + self.par.epsilon)

    def c1(self, l, p1, p2):
        """
        Calculate consumption of good 1.

        Parameters:
        l (float): Labor supplied by the household.
        p1 (float): Price of good 1.
        p2 (float): Price of good 2.

        Returns:
        float: Consumption of good 1.
        """
        profit1 = self.firm_profit1(p1)
        profit2 = self.firm_profit2(p2)
        return self.par.alpha * (self.w * l + self.par.T + profit1 + profit2) / p1

    def c2(self, l, p1, p2):
        """
        Calculate consumption of good 2.

        Parameters:
        l (float): Labor supplied by the household.
        p1 (float): Price of good 1.
        p2 (float): Price of good 2.

        Returns:
        float: Consumption of good 2.
        """
        profit1 = self.firm_profit1(p1)
        profit2 = self.firm_profit2(p2)
        return (1 - self.par.alpha) * (self.w * l + self.par.T + profit1 + profit2) / (p2 + self.par.tau)

    def optimize_l(self, p1, p2):
        """
        Optimize labor to maximize utility for workers.

        Parameters:
        p1 (float): Price of good 1.
        p2 (float): Price of good 2.

        Returns:
        tuple: Optimal labor, consumption of good 1, and consumption of good 2.
        """
        obj = lambda l: -self.utility_(l, p1, p2)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')
        l_star = res.x
        c1_star = self.c1(l_star, p1, p2)
        c2_star = self.c2(l_star, p1, p2)
        return l_star, c1_star, c2_star

    def firm_labor_demand(self, p1, p2):
        """
        Calculate firms' labor demand given prices.

        Parameters:
        p1 (float): Price of good 1.
        p2 (float): Price of good 2.

        Returns:
        tuple: Labor demand for firms producing goods 1 and 2.
        """
        l1_star = (p1 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        l2_star = (p2 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        return l1_star, l2_star

    def firm_profit1(self, p1):
        """
        Calculate profit for firm 1 given price.

        Parameters:
        p1 (float): Price of good 1.

        Returns:
        float: Profit for firm 1.
        """
        l1_star = (p1 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        y1_star = self.par.A * (l1_star)**self.par.gamma
        return (1 - self.par.gamma) * y1_star * p1

    def firm_profit2(self, p2):
        """
        Calculate profit for firm 2 given price.

        Parameters:
        p2 (float): Price of good 2.

        Returns:
        float: Profit for firm 2.
        """
        l2_star = (p2 * self.par.A * self.par.gamma / self.w)**(1 / (1 - self.par.gamma))
        y2_star = self.par.A * (l2_star)**self.par.gamma
        return (1 - self.par.gamma) * y2_star * p2

    def market_clearing_conditions(self, p):
        """
        Check market clearing conditions for labor and good 1.

        Parameters:
        p (list): List containing prices [p1, p2].

        Returns:
        list: Discrepancies in labor and good 1 markets.
        """
        p1, p2 = p
        l1_star, l2_star = self.firm_labor_demand(p1, p2)
        l_star, c1_star, c2_star = self.optimize_l(p1, p2)
        labor_market = l1_star + l2_star - l_star

        good_market_1 = c1_star - self.par.A * (l1_star)**self.par.gamma
        
        return [labor_market, good_market_1]
    
    def solve_market_clearing(self):
        """
        Solve for equilibrium prices p1 and p2.
        """
        def objective(p):
            return np.sum(np.square(self.market_clearing_conditions(p)))

        res = optimize.minimize(objective, [1, 1], bounds=((0.1, 10), (0.1, 10)))
        if res.success:
            self.p1, self.p2 = res.x
        else:
            raise ValueError("Market clearing optimization failed")

    def social_welfare(self, tau):
        """
        Calculate social welfare given a tax rate.

        Parameters:
        tau (float): Tax rate.

        Returns:
        float: Negative of the social welfare function.
        """
        self.par.tau = tau
        self.solve_market_clearing()  # Ensure we have equilibrium prices

        l_star, c1_star, c2_star = self.optimize_l(self.p1, self.p2)
        self.par.T = self.par.tau * c2_star  # Set T = tau * c2_star
        y2_star = self.par.A * (self.firm_labor_demand(self.p1, self.p2)[1])**self.par.gamma
        SWF = self.utility_(l_star, self.p1, self.p2) - self.par.kappa * y2_star
        return -SWF

    def optimize_tau(self):
        """
        Optimize the tax rate to maximize social welfare.

        Returns:
        tuple: Optimal tax rate and maximum social welfare.
        """
        result = optimize.minimize_scalar(lambda tau: -self.social_welfare(tau), bounds=(0, 2), method='bounded')
        optimal_tau = result.x
        max_swf = -result.fun
        self.par.tau = optimal_tau
        self.solve_market_clearing()  # Ensure prices are set for optimal tau
        return optimal_tau, max_swf




    