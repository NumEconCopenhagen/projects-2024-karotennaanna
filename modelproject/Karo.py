from types import SimpleNamespace
import numpy as np

class ISLMclass:

    def __init__(self, a, b, c, d, e, f, g, T, G, M, P):
        """ initialize the model """

        par = self.par = SimpleNamespace() # parameters
        sim = self.sim = SimpleNamespace() # simulation variables
        datamoms = self.datamoms = SimpleNamespace() # moments in the data
        moms = self.moms = SimpleNamespace() # moments in the model

        # Parameters of the model
        par.a = a # Autonomous private consumption
        par.b = b # The marginal propensity to consume out of disposable income
        par.c = c # Autonomous investment
        par.d = d # The sensitivity of investment to the real interest rate
        par.e = e # The sensitivity of real money demand to output
        par.f = f # The sensitivity of real money demand to the real interest rate
        par.g = g # Productivity

    def planned_expenditures(self):
        par = self.par
        return self.private_consumption() + self.investment_demand() + self.G

    def private_consumption(self):
        par = self.par
        return self.a + self.b * (self.Y() - self.T)

    def Y(self):
        par = self.par
        return self.planned_expenditures()

    def investment_demand(self):
        par = self.par
        return self.c - self.d * self.r

    def equilibrium_in_money_market(self):
        par = self.par
        return self.M / self.P == self.real_demand_for_money()

    def real_demand_for_money(self):
        par = self.par
        return self.e * self.Y() - self.f * self.r

    def production_function(self):
        par = self.par
        return self.g * self.N

    def IS_curve(self):
        par = self.par
        return (1 / self.d) * (self.a + self.c - self.b * self.T + self.G - (1 - self.b) * self.Y())
