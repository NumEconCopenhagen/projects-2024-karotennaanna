import sympy as sp

class ISLM_alg:
    def __init__(self):
        # Define symbols
        self.Y, self.r, self.a, self.b, self.c, self.d, self.e, self.f, self.T, self.G, self.M, self.P, self.PE, self.C, self.I, self.L = sp.symbols('Y r a b c d e f T G M P PE C I L')
        
        # Define equations
        self.PlannedExp = sp.Eq(self.PE, self.C + self.I + self.G)
        self.PrivateCons = sp.Eq(self.C, self.a + self.b * (self.Y - self.T))
        self.EqGoods = sp.Eq(self.Y, self.PE)
        self.Investment = sp.Eq(self.I, self.c - self.d * self.r)
        
        # Define the money market equations
        self.EqMoney = sp.Eq(self.M / self.P, self.L)
        self.dMoney = sp.Eq(self.L, self.e * self.Y - self.f * self.r)

    def derive_IS(self):
        # Substitute PE from PlannedExp into EqGoods
        EqGoods_sub = self.EqGoods.subs(self.PE, self.C + self.I + self.G)
        
        # Substitute the expression for C from PrivateCons into EqGoods_sub
        EqGoods_subsub = EqGoods_sub.subs(self.C, self.a + self.b * (self.Y - self.T))
        
        # Substitute I from Investment into EqGoods_subsub:
        EqGoods_final = EqGoods_subsub.subs(self.I, self.c - self.d * self.r)
        
        # Solve the system of equations for Y and r separately
        solution_Y = sp.solve(EqGoods_final, self.Y)
        solution_r = sp.solve(EqGoods_final, self.r)
        return solution_Y, solution_r

    def derive_LM(self):
        # Substitute EqMoney into dMoney
        dMoney_substituted = self.dMoney.subs(self.L, self.M / self.P)
        
        # Solve the LM equation for Y and r separately
        solution_Y = sp.solve(dMoney_substituted, self.Y)
        solution_r = sp.solve(dMoney_substituted, self.r)
        return solution_Y, solution_r
