import numpy as np

class CareerChoices:
    def simulate_initial_career_choices(par):
        # Step 1: Simulate ε_f,j for friends and ε_i,j for each graduate
        friends_epsilon = np.random.normal(0, par.sigma, (par.N, par.J, par.K))
        own_epsilon = np.random.normal(0, par.sigma, (par.N, par.J, par.K))

        # Initialize arrays to store results
        chosen_careers = np.zeros(par.N, dtype=int)
        expected_utilities = np.zeros(par.N)
        realized_utilities = np.zeros(par.N)

        # Step 2: Simulate and calculate the prior expected utility and choose the career with the highest expected utility
        for i in range(par.N):
            prior_expected_utility = np.zeros(par.J)

            # Calculate the prior expected utility based on friends' experiences
            for j in range(par.J):
                prior_expected_utility[j] = par.v[j] + np.mean(friends_epsilon[i, j, :par.F[i]])

            # Add the graduate's own noise term to the prior expected utility
            expected_utility_with_own_noise = np.zeros(par.J)
            for j in range(par.J):
                expected_utility_with_own_noise[j] = prior_expected_utility[j] + np.mean(own_epsilon[i, j, :par.K])

            # Choose the career with the highest expected utility
            chosen_career = np.argmax(expected_utility_with_own_noise)
            chosen_careers[i] = chosen_career

            # Store the expected and realized utilities for the chosen career
            expected_utilities[i] = prior_expected_utility[chosen_career]
            realized_utilities[i] = par.v[chosen_career] + np.mean(own_epsilon[i, chosen_career, :par.K])

        return chosen_careers, expected_utilities, realized_utilities, friends_epsilon, own_epsilon
    def simulate_second_career_choices(par, chosen_careers, realized_utilities, friends_epsilon, own_epsilon):
        # Initialize arrays to store results
        new_chosen_careers = np.zeros((par.N, par.K), dtype=int)
        new_expected_utilities = np.zeros((par.N, par.K))
        new_realized_utilities = np.zeros((par.N, par.K))
        switches = np.zeros((par.N, par.K), dtype=bool)

        # Step 2: Simulate and calculate the new expected utility considering switching costs
        for i in range(par.N):
            for k in range(par.K):
                prior_expected_utility = np.zeros(par.J)

                # Calculate the prior expected utility based on friends' experiences
                for j in range(par.J):
                    prior_expected_utility[j] = par.v[j] + friends_epsilon[i, j, k]

                # Modify the expected utilities to account for the switching cost
                new_prior_expected_utility = np.zeros(par.J)
                for j in range(par.J):
                    if j == chosen_careers[i]:
                        new_prior_expected_utility[j] = realized_utilities[i]
                    else:
                        new_prior_expected_utility[j] = prior_expected_utility[j] - par.c

                # Add the graduate's own noise term to the new prior expected utility
                new_expected_utility_with_own_noise = np.zeros(par.J)
                for j in range(par.J):
                    new_expected_utility_with_own_noise[j] = new_prior_expected_utility[j] + own_epsilon[i, j, k]

                # Choose the career with the highest expected utility
                new_chosen_career = np.argmax(new_expected_utility_with_own_noise)
                new_chosen_careers[i, k] = new_chosen_career

                # Store the expected and realized utilities for the new chosen career
                new_expected_utilities[i, k] = new_prior_expected_utility[new_chosen_career]
                if new_chosen_career == chosen_careers[i]:
                    new_realized_utilities[i, k] = realized_utilities[i]
                else:
                    new_realized_utilities[i, k] = par.v[new_chosen_career] + own_epsilon[i, new_chosen_career, k] - par.c

                # Check if the graduate switched careers
                switches[i, k] = new_chosen_career != chosen_careers[i]

        return new_chosen_careers, new_expected_utilities, new_realized_utilities, switches
model = CareerChoices()

class BarycentricInterpolation:
    def r1(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
        upper = (B[1] - C[1]) * (y1 - C[0]) + (C[0] - B[0]) * (y2 - C[1])
        lower = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = upper / lower
        return r   
    
    def r2(self, args, A, B, C):
        y1 = args[0]
        y2 = args[1]
        upper = (C[1] - A[1]) * (y1 - C[0]) + (A[0] - C[0]) * (y2 - C[1])
        lower = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r = upper / lower
        return r
    
    def r3(self, r1, r2):
        r = 1 - r1 - r2
        return r
    
    def distance(self, X, y):
        return np.sqrt((X[:, 0] - y[0])**2 + (X[:, 1] - y[1])**2)
    
    def A_distance(self, X, y):
        condition = X[(X[:, 0] > y[0]) & (X[:, 1] > y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def B_distance(self, X, y):
        condition = X[(X[:, 0] > y[0]) & (X[:, 1] < y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def C_distance(self, X, y):
        condition = X[(X[:, 0] < y[0]) & (X[:, 1] < y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def D_distance(self, X, y):
        condition = X[(X[:, 0] < y[0]) & (X[:, 1] > y[1])]
        if len(condition) == 0:
            return np.NaN
        distance_of_Xy = self.distance(condition, y)
        arg_min = np.argmin(distance_of_Xy)
        minimum = condition[arg_min]
        return minimum
    
    def isin_triangle(self, args, A, B, C):
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        inside = 0 <= r1val <= 1 and 0 <= r2val <= 1 and 0 <= r3val <= 1
        return inside
    
    def y_function(self, args, A, B, C):    
        r1val = self.r1(args, A, B, C)
        r2val = self.r2(args, A, B, C)
        r3val = self.r3(r1val, r2val)
        y = r1val * A + r2val * B + r3val * C
        return y

# Example usage
model = BarycentricInterpolation()
