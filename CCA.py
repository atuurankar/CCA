'''
Created on Jul 12, 2017

@author: Atulya
'''

import numpy as np
from scipy.optimize import minimize
import sys

class CCA():
    
    def __init__(self, x, y, x_penalty=None, x_l1_const=1, x_l2_const=1, 
                    y_penalty=None, y_l1_const=1, y_l2_const=1, constraints=None):
        
        # Input x and y
        x = np.array(x);
        y = np.array(y);
        
        # Defining penalty and its parameters for x vector
        self.x_penalty = x_penalty;
        self.x_l1_const = x_l1_const;
        self.x_l2_const = x_l2_const;
        
        # Defining penalty and its parameters for y vector
        self.y_penalty = y_penalty;
        self.y_l1_const = y_l1_const;
        self.y_l2_const = y_l2_const;
        
        # Account for any additional constraints on the CCA
        self.constraints = constraints;
        
        try:
            # Get shape of input vectors
            self.x_samples, self.x_dim = np.shape(x);
            self.y_samples, self.y_dim = np.shape(y);
            
            if self.x_samples != self.y_samples:
                print("Number of samples do not match in x and y");
                sys.exit();
            else:
                # Evaluate cross covariance matrices
                self.cov = np.cov(np.hstack((x, y)).T);
                self.cov_x = self.cov[:self.x_dim, :self.x_dim];
                self.cov_y = self.cov[self.x_dim:, self.x_dim:];
                self.cov_xy = self.cov[:self.x_dim, self.x_dim:];  
                  
        except NameError:
            print("Input vectors x and y incorrectly provided");
            sys.exit();

    # Define the objective function
    # "coeffs" will contain a stack of alpha and beta
    def _objective_function(self, coeffs):
        alpha = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        beta = np.reshape(coeffs[self.x_dim:], (self.y_dim, 1));
        
        # Setting parameters for penalty to be applied to x
        if self.x_penalty == None:
            x_l1_const = 0;
            x_l2_const = 0;
        elif self.x_penalty == "l1":
            x_l1_const = self.x_l1_const;
            x_l2_const = 0;
        elif self.x_penalty == "l2":
            x_l1_const = 0;
            x_l2_const = self.x_l2_const;
            
        # Setting parameters for penalty to be applied to y
        if self.y_penalty == None:
            y_l1_const = 0;
            y_l2_const = 0;
        elif self.y_penalty == "l1":
            y_l1_const = self.y_l1_const;
            y_l2_const = 0;
        elif self.y_penalty == "l2":
            y_l1_const = 0;
            y_l2_const = self.y_l2_const;
            
        return -np.matmul(np.matmul(alpha.T, self.cov_xy), beta) + \
                x_l1_const*np.sum(abs(alpha)) + x_l2_const*np.sqrt(np.sum(alpha**2)) + \
                y_l1_const*np.sum(abs(beta)) + y_l2_const*np.sqrt(np.sum(beta**2));
    
    # Define constraints for alpha such that variance alpha.T*x is normalized
    def _norm_alpha_var(self, coeffs):
        alpha = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        var = np.matmul(np.matmul(alpha.T, self.cov_x), alpha);
        return float(var) - 1;

    # Define constraints for beta such that variance beta.T*y is normalized
    def _norm_beta_var(self, coeffs):
        beta = np.reshape(coeffs[self.x_dim:], (self.y_dim, 1));
        var = np.matmul(np.matmul(beta.T, self.cov_y), beta);
        return float(var) - 1;
    
    # Define constraint for orthogonality of alpha
    def _orthonormal_alpha(self, coeffs, *alpha_i):
        alpha = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_i = np.reshape(np.array(alpha_i), (self.x_dim, 1));
        var = np.matmul(np.matmul(alpha_i.T, self.cov_x), alpha);
        return float(var);
        
    # Define constraint for orthogonality of beta
    def _orthonormal_beta(self, coeffs, *beta_i):
        beta = np.reshape(coeffs[self.x_dim:], (self.y_dim, 1));
        beta_i = np.reshape(np.array(beta_i), (self.y_dim, 1));
        var = np.matmul(np.matmul(beta_i.T, self.cov_y), beta);
        return float(var);   
    
    # Execute method to return canonical correlation coefficients
    def get_canonical_correlations(self):
        
        # Minimum of x and y dimension to get number of canonical correlations
        min_x_y = min(self.x_dim, self.y_dim);
        
        # Define an array of all constraints
        constraints = np.array([]).astype(object);
        if self.constraints != None:
            constraints = np.append(constraints, self.constraints);
        constraints = np.append(constraints, {"type": "eq", "fun": self._norm_alpha_var});
        constraints = np.append(constraints, {"type": "eq", "fun": self._norm_beta_var});
        
        # Initialize an array containing values of alpha vector for all canonical correlations
        self.alpha = np.zeros((min_x_y, self.x_dim));
        
        # Initialize an array containing values of alpha vector for all canonical correlations
        self.beta = np.zeros((min_x_y, self.y_dim));
        
        for i in range(min_x_y):
            temp_constraints = np.array(constraints);
            for j in range(i):
                temp_constraints = np.append(temp_constraints, {"type": "eq", "fun": self._orthonormal_alpha, "args": self.alpha[j]});
                temp_constraints = np.append(temp_constraints, {"type": "eq", "fun": self._orthonormal_beta, "args": self.beta[j]});
                
            output = minimize(fun=self._objective_function,
                              constraints=temp_constraints,
                              x0=np.reshape(np.random.rand(self.x_dim + self.y_dim), (self.x_dim + self.y_dim, 1)),
                              method="SLSQP")["x"];
            
            if not output["success"]:
                print("Numerical Optimization did not converge");
                sys.exit();

            coeffs = output["x"];              
            self.alpha[i], self.beta[i] = coeffs[:self.x_dim], coeffs[self.x_dim:];   
         
        return self.alpha, self.beta;
    