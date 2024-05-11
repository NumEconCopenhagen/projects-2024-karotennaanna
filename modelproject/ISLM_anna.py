class ISLMclass:
    def __init__(self, a, b, c, d, e, f, g, T, G, M, P):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.T = T
        self.G = G
        self.M = M
        self.P = P

    def IS_curve(self, Y, r):
        return (1 / self.d) * (self.a + self.c - self.b * self.T + self.G - (1 - self.b) * Y) - Y

    def LM_curve(self, Y, r):
        return (self.e / self.f) * Y - (1 / self.f) * (self.M / self.P) - r

    def system_equations(self, vars):
        Y, r = vars
        return [self.IS_curve(Y, r), self.LM_curve(Y, r)]


