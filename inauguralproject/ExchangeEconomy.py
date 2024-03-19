from types import SimpleNamespace
from scipy import optimize
import numpy as np

class ExchangeEconomyClass:
    
    def __init__(self):

        self.par = SimpleNamespace()

        # a. preferences
        self.par.alpha = 1/3
        self.par.beta = 2/3

        # b. endowments
        self.par.w1A = 0.8
        self.par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        return x1A**self.par.alpha * x2A**(1-self.par.alpha)

    def utility_B(self,x1B,x2B):
        return x1B**self.par.beta * x2B**(1-self.par.beta) 

    def demand_A(self,p1):
        x1A = self.par.alpha*(p1*self.par.w1A+self.par.w2A)/p1
        x2A = (1-self.par.alpha)*(p1*self.par.w1A+self.par.w2A)

        return x1A, x2A

    def demand_B(self,p1):
        x1B = self.par.beta*(p1*self.par.w1A+self.par.w2A)/p1
        x2B = (1-self.par.beta)*(p1*self.par.w1A+self.par.w2A)

        return x1B, x2B
    
    def check_market_clearing(self,p1):

        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        eps1 = x1A - self.par.w1A + x1B - (1 - self.par.w1A)
        eps2 = x2A - self.par.w2A + x2B - (1 - self.par.w2A)

        return eps1, eps2
    
    def question1(self):
        self.list_x1A = []
        self.list_x2A = []

        for j in range(76):  # Including 75
            x1A = j / 75
            for i in range(76):  # Including 75
                x2A = i / 75
                x1B = 1 - x1A
                x2B = 1 - x2A

                if self.utility_A(x1A, x2A) >= self.utility_A(self.par.w1A, self.par.w2A) and \
                   self.utility_B(x1B, x2B) >= self.utility_B(1 - self.par.w1A, 1 - self.par.w2A):
                    self.list_x1A.append(x1A)
                    self.list_x2A.append(x2A)

    def question2(self):
        self.price_list = []
        self.error1 = []
        self.error2 = []
        
        for var in range(76):
            price = 0.5 + 2 * (var / 75)
            eps1, eps2 = self.check_market_clearing(price)
            
            self.price_list.append(price)
            self.error1.append(eps1)
            self.error2.append(eps2)
    
    def market_clearing_error(self,p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)

        eps1 = x1A - self.par.w1A + x1B - (1 - self.par.w1A)
        eps2 = x2A - self.par.w2A + x2B - (1 - self.par.w2A)

        return eps1+eps2
        