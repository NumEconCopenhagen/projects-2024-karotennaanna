import numpy as np

class ProductionEconomyModel:
    def __init__(self, A=1.0, gamma=0.5, alpha=0.3, nu=1.0, epsilon=2.0, tau=0.0, T=0.0):
        """
        Initializes the Production Economy Model with given parameters.
        :param A: Productivity parameter for firms.
        :param gamma: Elasticity parameter for production function.
        :param alpha: Consumer preference parameter for good 1.
        :param nu: Disutility of labor parameter.
        :param epsilon: Labor supply elasticity parameter.
        :param tau: Tax on good 2.
        :param T: Lump-sum transfer.
        """
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T

    def labor_supply(self, w, p1, p2):
        """
        Computes optimal labor supply given wages and prices.
        :param w: Wage rate.
        :param p1: Price of good 1.
        :param p2: Price of good 2.
        :return: Optimal labor supply.
        """
        term = w * (p1**self.alpha) * ((p2 + self.tau)**(1 - self.alpha))
        labor = (term / self.nu)**(1 / (self.epsilon + 1))
        return labor

    def production(self, w, p1, p2):
        """
        Computes optimal production level for each firm given wages and prices.
        :param w: Wage rate.
        :param p1: Price of good 1.
        :param p2: Price of good 2.
        :return: Tuple of optimal production levels for each good.
        """
        labor = self.labor_supply(w, p1, p2)
        production1 = self.A * (labor**self.gamma)
        production2 = self.A * (labor**self.gamma)
        return production1, production2

    def consumer_utility(self, w, p1, p2):
        """
        Computes the utility of the consumer.
        :param w: Wage rate.
        :param p1: Price of good 1.
        :param p2: Price of good 2.
        :return: Consumer utility.
        """
        labor = self.labor_supply(w, p1, p2)
        income = w * labor + self.T
        consumption1 = self.alpha * income / p1
        consumption2 = (1 - self.alpha) * income / (p2 + self.tau)
        utility = np.log(consumption1**self.alpha * consumption2**(1 - self.alpha)) - self.nu * labor**(1 + self.epsilon) / (1 + self.epsilon)
        return utility

    def check_market_clearing(self, w, p1, p2):
        """
        Checks the market clearing conditions for labor and goods.
        :param w: Wage rate.
        :param p1: Price of good 1.
        :param p2: Price of good 2.
        :return: Dictionary indicating if labor and goods markets clear.
        """
        labor = self.labor_supply(w, p1, p2)
        production1, production2 = self.production(w, p1, p2)
        consumer_labor = labor
        consumer_goods1 = self.alpha * (w * labor + self.T) / p1
        consumer_goods2 = (1 - self.alpha) * (w * labor + self.T) / (p2 + self.tau)
        
        return {
            'Labor Market': np.isclose(labor, consumer_labor),
            'Good Market 1': np.isclose(production1, consumer_goods1),
            'Good Market 2': np.isclose(production2, consumer_goods2)
        }


