import numpy as np

def optimal_labor_supply(tau, w, G, alpha=0.5, kappa=1.0, nu=1/(2 * 162)):
    """
    Calculate the optimal labor supply given parameters.
    
    Parameters:
    tau (float): Tax rate
    w (float): Real wage
    G (float): Government consumption
    alpha (float): Weight for private consumption
    kappa (float): Free private consumption component
    nu (float): Disutility of labor scaling factor
    
    Returns:
    float: Optimal labor supply
    """
    # Calculate tilde_w
    tilde_w = (1 - tau) * w / G
    # Calculate the discriminant of the square root
    discriminant = kappa**2 + 4 * tilde_w**2 * alpha / nu
    # Calculate the optimal labor supply using the given formula
    optimal_L = (-kappa + np.sqrt(discriminant)) / (2 * tilde_w)
    return optimal_L



