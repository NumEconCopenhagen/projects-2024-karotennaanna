from types import SimpleNamespace
from scipy.optimize import minimize, minimize_scalar
import numpy as np
import matplotlib.pyplot as plt

class ExchangeEconomyClass:
    def __init__(self):
        self.par = SimpleNamespace(alpha=1/3, beta=2/3, w1A=0.8, w2A=0.3, w1B=0.2, w2B=0.7)

    def utility(self, x1, x2, alpha):
        # Ensure no invalid operations
        if x1 <= 0 or x2 <= 0:
            return -np.inf
        return x1 ** alpha * x2 ** (1 - alpha)

    def utility_A(self, x1A, x2A):
        return self.utility(x1A, x2A, self.par.alpha)

    def utility_B(self, x1B, x2B):
        return self.utility(x1B, x2B, self.par.beta)

    def demand_A(self, p1):
        x1A = self.par.alpha * (self.par.w1A + self.par.w2A / p1)
        x2A = (1 - self.par.alpha) * (p1 * self.par.w1A + self.par.w2A)
        return x1A, x2A

    def demand_B(self, p1):
        x1B = self.par.beta * (self.par.w1B + self.par.w2B / p1)
        x2B = (1 - self.par.beta) * (p1 * self.par.w1B + self.par.w2B)
        return x1B, x2B

    def check_market_clearing(self, p1):
        x1A, x2A = self.demand_A(p1)
        x1B, x2B = self.demand_B(p1)
        eps1 = x1A + x1B - 1
        eps2 = x2A + x2B - 1
        return eps1, eps2

    def market_clearing_error(self, p1):
        eps1, eps2 = self.check_market_clearing(p1)
        return abs(eps1) + abs(eps2)

    def find_market_clearing_price(self):
        result = minimize_scalar(self.market_clearing_error, bounds=(0.5, 2.5), method='bounded')
        return result.x

    # Method for question 4a
    def find_optimal_price_for_A_in_P1(self):
        def utility_A_neg(p1):
            x1B, x2B = self.demand_B(p1)
            return -self.utility_A(1 - x1B, 1 - x2B)

        result = minimize_scalar(utility_A_neg, bounds=(0.5, 2.5), method='bounded')
        optimal_price_p1 = result.x
        optimal_x1B, optimal_x2B = self.demand_B(optimal_price_p1)
        return optimal_price_p1, (1 - optimal_x1B, 1 - optimal_x2B)

    # Method for question 4b
    def find_optimal_price_for_A_unbounded(self):
        def utility_A_neg(p1):
            x1B, x2B = self.demand_B(p1)
            return -self.utility_A(1 - x1B, 1 - x2B)

        result = minimize_scalar(utility_A_neg, bounds=(0.5, 2.5), method='bounded')
        optimal_price_p1 = result.x
        optimal_x1B, optimal_x2B = self.demand_B(optimal_price_p1)
        return optimal_price_p1, (1 - optimal_x1B, 1 - optimal_x2B)

    def find_optimal_allocation_for_A(self, x1_possible, x2_possible):
        uAmax = -np.inf
        x1best = -np.inf
        x2best = -np.inf

        for x1, x2 in zip(x1_possible, x2_possible):
            uAnew = self.utility_A(x1, x2)
            if uAnew > uAmax and self.utility_B(1 - x1, 1 - x2) >= self.utility_B(1 - self.par.w1A, 1 - self.par.w2A):
                uAmax = uAnew
                x1best = x1
                x2best = x2

        return uAmax, x1best, x2best

    def find_optimal_allocation_for_A_with_constraints(self, w1A, w2A):
        def utility_A_neg(x):
            return -self.utility_A(x[0], x[1])

        def constraint_B(x):
            xB1 = 1 - x[0]
            xB2 = 1 - x[1]
            return self.utility_B(xB1, xB2) - self.utility_B(1 - w1A, 1 - w2A)

        initial_guess = [0.5, 0.5]
        result = minimize(utility_A_neg, initial_guess, bounds=[(0, 1), (0, 1)],
                          constraints={'type': 'ineq', 'fun': constraint_B})
        return result.x

    def find_optimal_allocation_to_maximize_aggregate_utility(self):
        def aggregate_utility(x):
            x1A, x2A = x
            x1B, x2B = 1 - x1A, 1 - x2A
            return -(self.utility_A(x1A, x2A) + self.utility_B(x1B, x2B))

        result = minimize(aggregate_utility, [0.5, 0.5], bounds=[(0, 1), (0, 1)])
        optimal_x1A, optimal_x2A = result.x
        return optimal_x1A, optimal_x2A

    def find_contract_curve(self):
        contract_allocations = []
        for x1A in np.linspace(0, 1, 100):
            def objective(x):
                x2A = x[0]
                x1B = 1 - x1A
                x2B = 1 - x2A
                return -(self.utility_A(x1A, x2A) + self.utility_B(x1B, x2B))

            cons = ({
                'type': 'eq',
                'fun': lambda x: self.par.alpha / (1 - self.par.alpha) * (1 - x1A) / x[0] - self.par.beta / (1 - self.par.beta) * x1A / (1 - x[0])
            })

            result = minimize(objective, [0.5], bounds=[(0, 1)], constraints=cons)
            optimal_x2A = result.x[0]
            contract_allocations.append((x1A, optimal_x2A))
        
        return np.array(contract_allocations)

class ExchangeEconomyPlotter:
    def __init__(self, model, N=75, w1A=0.8, w2A=0.3):
        self.model = model
        self.N = N
        self.w1A = w1A
        self.w2A = w2A
        self.w1bar = 1.0
        self.w2bar = 1.0

    def calculate_possible_values(self):
        x1A = np.linspace(0, 1, self.N + 1)
        x1_possible, x2_possible = [], []
        for x1 in x1A:
            for x2 in x1A:
                if (self.model.utility_A(x1, x2) >= self.model.utility_A(self.w1A, self.w2A) and
                        self.model.utility_B(1 - x1, 1 - x2) >= self.model.utility_B(1 - self.w1A, 1 - self.w2A)):
                    x1_possible.append(x1)
                    x2_possible.append(x2)
        return x1_possible, x2_possible

    def plot_pareto_improvements(self):
        x1_possible, x2_possible = self.calculate_possible_values()
        par = self.model.par

        fig, ax_A = plt.subplots(figsize=(6, 6), dpi=100)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        ax_B = ax_A.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        temp.invert_yaxis()

        ax_A.scatter(par.w1A, par.w2A, marker='s', color='red', label='endowment')
        ax_A.scatter(x1_possible, x2_possible, alpha=0.5, label='pareto improvements')
        self._plot_limits(ax_A, ax_B, temp)

        ax_A.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.76, 0.9))
        plt.show()

    def plot_market_clearing_errors(self):
        p1 = [0.5 + 2 * i / self.N for i in range(self.N + 1)]
        error = [self.model.check_market_clearing(price) for price in p1]
        error1, error2 = zip(*error)

        fig, ax_C = plt.subplots(figsize=(6, 6), dpi=100)
        ax_C.set_ylabel("Error under market clearing")
        ax_C.set_xlabel("$p_1$")
        ax_C.set_title("Market Clearing errors under $\mathcal{P}_1$")
        ax_C.plot(p1, error1, label='$\epsilon_1(p,\omega)$')
        ax_C.plot(p1, error2, label='$\epsilon_2(p,\omega)$')
        ax_C.legend()
        plt.show()

    def _plot_limits(self, ax_A, ax_B, temp):
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        temp.set_ylim([self.w2bar + 0.1, -0.1])

class ExchangeEconomyVisualizer:
    def __init__(self):
        self.w1bar = 1.0
        self.w2bar = 1.0

    def plot_points(self, points, point_labels, colors):
        fig, ax_A = plt.subplots(figsize=(6, 6), dpi=100)
        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")
        ax_B = ax_A.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        temp.invert_yaxis()

        self._plot_limits(ax_A, ax_B, temp)
        for point, label, color in zip(points, point_labels, colors):
            ax_A.scatter(*point, color=color, label=label)

        ax_A.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.4, 0.6))
        plt.show()

    def plot_random_set(self, num_elements):
        w1A = np.random.rand(num_elements)
        w2A = np.random.rand(num_elements)

        plt.figure(figsize=(8, 6))
        plt.scatter(w1A, w2A, color='b', marker='o')
        plt.title('Random Set W')
        plt.xlabel('w1A')
        plt.ylabel('w2A')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()

    def plot_estimated_equilibrium_allocations(self, contract_allocations):
        plt.figure(figsize=(8, 8))
        plt.scatter(contract_allocations[:, 0], contract_allocations[:, 1], color='blue', label='Contract Curve')
        plt.title('Estimated Equilibrium Allocations in Edgeworth Box')
        plt.xlabel('$x_1^A$ (Good 1 Allocation to A)')
        plt.ylabel('$x_2^A$ (Good 2 Allocation to A)')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axline((0, 0), (1, 1), color='red', linestyle='--', label='Tot. Endowments Line')
        plt.legend(frameon=True, loc='lower right', bbox_to_anchor=(1.5, 0.9))
        plt.grid(True)
        plt.show()

    def _plot_limits(self, ax_A, ax_B, temp):
        ax_A.plot([0, self.w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, self.w1bar], [self.w2bar, self.w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, self.w2bar], lw=2, color='black')
        ax_A.plot([self.w1bar, self.w1bar], [0, self.w2bar], lw=2, color='black')
        ax_A.set_xlim([-0.1, self.w1bar + 0.1])
        ax_A.set_ylim([-0.1, self.w2bar + 0.1])
        ax_B.set_xlim([self.w1bar + 0.1, -0.1])
        temp.set_ylim([self.w2bar + 0.1, -0.1])

# Instantiate the economic model
model = ExchangeEconomyClass()

# Question 4a
optimal_price_p1_4a, optimal_allocation_A_4a = model.find_optimal_price_for_A_in_P1()
print("Optimal price for 4a:", optimal_price_p1_4a)
print("Optimal allocation for A (4a):", optimal_allocation_A_4a)

# Question 4b
optimal_price_p1_4b, optimal_allocation_A_4b = model.find_optimal_price_for_A_unbounded()
print("Optimal price for 4b:", optimal_price_p1_4b)
print("Optimal allocation for A (4b):", optimal_allocation_A_4b)


























