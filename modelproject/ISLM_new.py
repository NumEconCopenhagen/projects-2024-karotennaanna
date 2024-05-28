import sympy as sp

class ISLM_alg:
    def __init__(self, a, b, c, d, e, f, T, G, M, P):
        # Assign parameters to instance variables
        self.a, self.b, self.c, self.d, self.e, self.f, self.T, self.G, self.M, self.P = a, b, c, d, e, f, T, G, M, P
        
        # Define symbols
        self.Y, self.r = sp.symbols('Y r')
        self.C, self.I, self.PE, self.L = sp.symbols('C I PE L')
        
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
        
        return EqGoods_final

    def derive_LM(self):
        # Substitute EqMoney into dMoney
        dMoney_substituted = self.dMoney.subs(self.L, self.M / self.P)
        
        return dMoney_substituted
    
    def solve_for_Y(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_Y = sp.solve(IS_eq, self.Y)
        LM_solution_Y = sp.solve(LM_eq, self.Y)
        return IS_solution_Y, LM_solution_Y

    def solve_for_r(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        IS_solution_r = sp.solve(IS_eq, self.r)
        LM_solution_r = sp.solve(LM_eq, self.r)
        return IS_solution_r, LM_solution_r

    def find_equilibrium(self):
        IS_eq = self.derive_IS()
        LM_eq = self.derive_LM()
        
        # Solve the IS and LM equations together for Y and r
        equilibrium = sp.solve([IS_eq, LM_eq], (self.Y, self.r), dict=True)
        
        return equilibrium
    
    def objective(self, params):
        T, G, M = params
        self.T, self.G, self.M = T, G, M
        
        equilibrium = self.find_equilibrium()
        if equilibrium:
            for sol in equilibrium:
                if self.r in sol:
                    return (sol[self.r] - 0.04)**2
        return float('inf')

    def optimize_parameters(self):
        initial_guess = [self.T, self.G, self.M]
        result = minimize(self.objective, initial_guess, method='Nelder-Mead')
        self.T, self.G, self.M = result.x
        return result.x