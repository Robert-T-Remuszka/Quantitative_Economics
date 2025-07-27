import numpy as np
from scipy.optimize import minimize_scalar, bisect

'''def u(c,γ):
    return c ** (1-γ)/ (1-γ)

def v_star(x, β, γ):
    return (1 - β**(1/γ))**(-γ) * u(x,γ)

def c_star(x, β, γ):
    return (1 - β**(1/γ)) * x'''

def maximize(g, a, b, *args):
    '''
    Maximize the function g over the interval [a,b]. The *args tuple is used to calculate
    v(x-c). See definition of the methods g.
    '''
    objective = lambda c: -g(c, *args)
    res = minimize_scalar(objective, bounds = (a, b), method = 'bounded')
    argmax, maximum = res.x, -res.fun
    return argmax, maximum

class CakeEating:

    def __init__(self,
                 β = 0.96,
                 γ = 1.5,
                 x_min = 1e-3,
                 x_max = 2.5,
                 n = 200,
                 tol = 1e-4,
                 max_iter = 1000):
        
        # Set parameter (attribute) values according to user
        self.β, self.γ = β, γ

        # Initialize state space and equilibrium objects
        self.x_grid= np.linspace(x_min, x_max, n)
        self.v0  = np.zeros(n) 
        self.tol, self.max_iter  = tol, max_iter

    def u(self, c):

        '''
        Utility function.
        '''
        # Convenient shorthand for self.\gamma
        γ = self.γ

        if γ == 1:
            return np.log(c)
        else:
            return c**(1-γ)/(1-γ)
        
    def u_prime(self, c):

        '''
        Marginal utility.
        '''
        γ = self.γ
        return c**(-γ)
    
    def g(self, c, x, v_array):

        '''
        The objective function inside the Bellman operator.
        '''
        u, β = self.u, self.β
        v = lambda x: np.interp(x, self.x_grid, v_array)
        
        return u(c) + β * v(x - c)
    
    def T(self, v):

        '''
        The Bellman Opeerator
        '''
        g = self.g
        x_grid = self.x_grid
        v_new = np.empty_like(v)

        # Store the maximum attainable value at each grid point
        for i, x in enumerate(x_grid):
            v_new[i] = maximize(g, 1e-10, x, x, v)[1]

        return v_new
    
    def T_iter(self, verbose = True, print_skip = 25):

        '''
        Solve Bellman by value function iteration.
        '''
        tol = self.tol
        max_iter = self.max_iter
        error = 1 + tol
        i = 0
        v = self.v0
        
        while error > tol and i < max_iter:

            v_new = self.T(v)
            error = np.max(np.abs(v - v_new))
            i += 1

            if verbose and i % print_skip == 0:
                print(f"Error at iteration {i} is {error}.")

            v = v_new
        
        if error > tol:
            print("Failed to converge.")
        elif verbose:
            print(f"Converged in {i} iterations.")
        
        return v_new 


    
    def v_star(self, x):

        '''
        Exact solution.
        '''
        u, β, γ = self.u, self.β, self.γ

        return (1 - β**(1/γ))**(-γ) * u(x)


    

        