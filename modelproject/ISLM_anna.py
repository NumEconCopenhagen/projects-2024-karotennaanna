from types import SimpleNamespace

class ISLMclass:
    def __init__(self, a, b, c, d, e, f, g, T, G, M, P, r, N):
        self.par = SimpleNamespace(a=a, b=b, c=c, d=d, e=e, f=f, g=g, T=T, G=G, M=M, P=P, r=r, N=N)

    def planned_expenditures(self, Y):
        # Calculate planned expenditures directly using Y
        return self.private_consumption(Y) + self.investment_demand() + self.par.G

    def private_consumption(self, Y):
        # Calculate private consumption using Y
        return self.par.a + self.par.b * (Y - self.par.T)

    def investment_demand(self):
        # Calculate investment demand using the class interest rate
        return self.par.c - self.par.d * self.par.r

    def real_demand_for_money(self, Y):
        # Calculate real demand for money using Y
        return self.par.e * Y - self.par.f * self.par.r

    def equilibrium_in_money_market(self, Y):
        # Equation for money market equilibrium
        return self.par.M / self.par.P - self.real_demand_for_money(Y)

    def IS_curve(self, Y):
        # IS curve defined explicitly with Y
        return (1 / self.par.d) * (self.par.a + self.par.c - self.par.b * self.par.T + self.par.G - (1 - self.par.b) * Y) - Y

    def LM_curve(self, Y):
        # LM curve defined explicitly with Y
        return (self.par.e / self.par.f) * Y - (1 / self.par.f) * (self.par.M / self.par.P)


class ISLMfclass:
    def __init__(self, a, b, c, d, e, f, g, T, G, M, P, r, N, X, IM, e_rate, i_foreign):
        # X is the export responsiveness to exchange rate, IM is the import function
        self.par = SimpleNamespace(a=a, b=b, c=c, d=d, e=e, f=f, g=g, T=T, G=G, M=M, P=P, r=r, N=N, X=X, IM=IM, e_rate=e_rate, i_foreign=i_foreign)

    def planned_expenditures(self, Y):
        return self.private_consumption(Y) + self.investment_demand() + self.par.G + self.net_exports(Y)

    def private_consumption(self, Y):
        return self.par.a + self.par.b * (Y - self.par.T)

    def investment_demand(self):
        return self.par.c - self.par.d * self.par.r

    def net_exports(self, Y):
        # Net exports are a function of exchange rate and foreign demand
        return self.par.X * self.par.e_rate - self.par.IM * Y

    def real_demand_for_money(self, Y):
        # Adjusted for foreign interest rates
        return self.par.e * Y - self.par.f * (self.par.r - self.par.i_foreign)

    def equilibrium_in_money_market(self, Y):
        return self.par.M / self.par.P - self.real_demand_for_money(Y)

    def IS_curve(self, Y):
        return self.planned_expenditures(Y) - Y

    def LM_curve(self, Y):
        return (self.par.e / self.par.f) * Y - (1 / self.par.f) * (self.par.M / self.par.P)
