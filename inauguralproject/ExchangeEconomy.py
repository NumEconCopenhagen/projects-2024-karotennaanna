from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        par.p1 = 0 

    def utility_A(self,x1A,x2A):
        return x1A**alpha*x2A**(1-alpha)

    def utility_B(self,x1B,x2B):
        return x1B**beta*x2B**(1-beta)

    def demand_A(self,p1):
        return alpha*((p1*w1A+par.w2A)/P1), (1-alpha)*((p1*w1A+par.w2A)/P1)

    def demand_B(self,p1):
        return beta*((p1*w1B+w2B)/P1), (1-beta)*((p1*w1B+w2B)/P1)

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2