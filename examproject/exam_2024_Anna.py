import numpy as np
from types import SimpleNamespace

class EconomicModel:
    def __init__(self, A, gamma, alpha, nu, epsilon, tau=0.0, T=0.0):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T

    def firm_behavior(self, w, p):
        l_star = (p * self.A * self.gamma / w)**(1/(1 - self.gamma))
        y_star = self.A * (l_star**self.gamma)
        pi_star = w * l_star * (1 - self.gamma) / self.gamma
        return l_star, y_star, pi_star

    def consumer_behavior(self, p1, p2, w, T, pi1, pi2):
        c1_star = self.alpha * (w + T + pi1 + pi2) / p1
        c2_star = (1 - self.alpha) * (w + T + pi1 + pi2) / (p2 + self.tau)
        l_star = ((w + T + pi1 + pi2) * self.nu / (p1*c1_star + (p2 + self.tau)*c2_star))**(1/(1 + self.epsilon))
        return c1_star, c2_star, l_star

    def market_clearing(self, w, p1, p2):
        l1, y1, pi1 = self.firm_behavior(w, p1)
        l2, y2, pi2 = self.firm_behavior(w, p2)
        c1, c2, l_star = self.consumer_behavior(p1, p2, w, self.T, pi1, pi2)
        labor_clearing = (l1 + l2) - l_star
        goods_clearing1 = y1 - c1
        goods_clearing2 = y2 - c2
        return labor_clearing, goods_clearing1, goods_clearing2
