'''
Created on Jul 26, 2017

@author: Atulya
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys

np.set_printoptions(precision=3);

class CCA():
    
    def __init__(self, x, y, x_penalty=None, x_l1_const=1, x_l2_const=1, 
                    y_penalty=None, y_l1_const=1, y_l2_const=1, constraints=None):
        
        # Input x and y
        x = np.array(x);
        y = np.array(y);
        
        # Check if the input arrays contain nan
        if pd.isnull(x).sum() > 0:
            print("Input array x contains nan values");
            sys.exit();
        elif pd.isnull(y).sum() > 0:
            print("Input array y contains nan values");
            sys.exit();
        
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
        
        x_samples = 0;
        y_samples = 0;
        try:
            # Get shape of input vectors
            try:
                x_samples, self.x_dim = np.shape(x);
            # Error may be raised if 1 dimensional vector is passed
            except ValueError:
                x_samples = np.shape(x)[0];
                self.x_dim = 1;
                x = np.reshape(x, (x_samples, self.x_dim));
            try:
                y_samples, self.y_dim = np.shape(y);
            # Error may be raised if 1 dimensional vector is passed
            except ValueError:
                y_samples = np.shape(y)[0];
                self.y_dim = 1;
                y = np.reshape(y, (y_samples, self.y_dim));
            
            if x_samples != y_samples:
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
        alpha_pos = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_neg = np.reshape(coeffs[self.x_dim: self.x_dim + self.x_dim], (self.x_dim, 1));
        beta = np.reshape(coeffs[self.x_dim + self.x_dim:], (self.y_dim, 1));
        
        # Initialize Objective function
        objective_func = -np.matmul(np.matmul((alpha_pos - alpha_neg).T, self.cov_xy), beta);
        
        # Setting parameters for penalty to be applied to x
        if self.x_penalty == "l1":
            x_l1_const = self.x_l1_const;
            objective_func += x_l1_const*np.sum(abs((alpha_pos - alpha_neg)));
        elif self.x_penalty == "l2":
            x_l2_const = self.x_l2_const;
            objective_func += x_l2_const*np.sqrt(np.sum((alpha_pos - alpha_neg)**2))
            
        # Setting parameters for penalty to be applied to y
        if self.y_penalty == "l1":
            y_l1_const = self.y_l1_const;
            objective_func += y_l1_const*np.sum(abs(beta));
        elif self.y_penalty == "l2":
            y_l2_const = self.y_l2_const;
            objective_func += y_l2_const*np.sqrt(np.sum(beta**2))
            
        return objective_func;
    
    # Define the jacobian for the objective function
    # temporarily meant to work without any regularization
    def _objective_jacobian(self, coeffs):
        alpha_pos = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_neg = np.reshape(coeffs[self.x_dim: self.x_dim + self.x_dim], (self.x_dim, 1));        
        beta = np.reshape(coeffs[self.x_dim + self.x_dim:], (self.y_dim, 1));  
        
        grad_alpha_pos = -np.matmul(beta.T, self.cov_xy.T);
        grad_alpha_neg = np.matmul(beta.T, self.cov_xy.T);
        grad_beta = -np.matmul((alpha_pos - alpha_neg).T, self.cov_xy);
        return np.hstack((grad_alpha_pos, grad_alpha_neg, grad_beta));
    
    # Define constraints and its jacobian for alpha such that variance alpha.T*x is normalized
    # "coeffs" will contain a stack of alpha and beta
    def _norm_alpha_var(self, coeffs):
        alpha_pos = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_neg = np.reshape(coeffs[self.x_dim: self.x_dim + self.x_dim], (self.x_dim, 1)); 
        var = np.matmul(np.matmul((alpha_pos - alpha_neg).T, self.cov_x), (alpha_pos - alpha_neg));
        return float(var) - 1;

    def _norm_alpha_var_constraint_jacobian(self, coeffs):
        alpha_pos = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_neg = np.reshape(coeffs[self.x_dim: self.x_dim + self.x_dim], (self.x_dim, 1)); 
        grad_alpha_pos = np.matmul(alpha_pos.T, (self.cov_x + self.cov_x.T)) - \
                         np.matmul(alpha_neg.T, self.cov_x) - np.matmul(alpha_neg.T, self.cov_x.T);
        grad_alpha_neg = -np.matmul(alpha_pos.T, self.cov_x.T) - np.matmul(alpha_pos.T, self.cov_x) + \
                         np.matmul(alpha_neg.T, (self.cov_x + self.cov_x.T));
        return np.hstack((grad_alpha_pos, grad_alpha_neg, np.tile(0, (1, self.y_dim))));

    # Define constraints and its jacobian for beta such that variance beta.T*y is normalized
    # "coeffs" will contain a stack of alpha and beta
    def _norm_beta_var(self, coeffs):
        beta = np.reshape(coeffs[self.x_dim + self.x_dim:], (self.y_dim, 1));
        var = np.matmul(np.matmul(beta.T, self.cov_y), beta);
        return float(var) - 1;
    
    def _norm_beta_var_constraint_jacobian(self, coeffs):
        beta = np.reshape(coeffs[self.x_dim + self.x_dim:], (self.y_dim, 1));
        grad_beta = np.matmul(beta.T, (self.cov_y + self.cov_y.T));
        return np.hstack((np.tile(0, (1, self.x_dim + self.x_dim)), grad_beta));
    
    # Define constraint and jacobian for orthogonality of alpha
    # "coeffs" will contain a stack of alpha and beta
    def _orthonormal_alpha_constraint(self, coeffs, *alpha_i):
        alpha_pos = np.reshape(coeffs[:self.x_dim], (self.x_dim, 1));
        alpha_neg = np.reshape(coeffs[self.x_dim: self.x_dim + self.x_dim], (self.x_dim, 1));
        alpha_i = np.reshape(np.array(alpha_i), (self.x_dim, 1));
        var = np.matmul(np.matmul(alpha_i.T, self.cov_x), (alpha_pos - alpha_neg));
        return float(var);
    
    def _orthonormal_alpha_constraint_jacobian(self, coeffs, *alpha_i):
        alpha_i = np.reshape(np.array(alpha_i), (self.x_dim, 1));
        grad_alpha_pos = np.matmul(alpha_i.T, self.cov_x.T);
        grad_alpha_neg = -np.matmul(alpha_i.T, self.cov_x.T);
        return np.hstack((grad_alpha_pos, grad_alpha_neg, np.tile(0, (1, self.y_dim))));
        
    # Define constraint and jacobian for orthogonality of beta
    # "coeffs" will contain a stack of alpha and beta
    def _orthonormal_beta_constraint(self, coeffs, *beta_i):
        beta = np.reshape(coeffs[self.x_dim + self.x_dim:], (self.y_dim, 1));
        beta_i = np.reshape(np.array(beta_i), (self.y_dim, 1));
        var = np.matmul(np.matmul(beta_i.T, self.cov_y), beta);
        return float(var);   
    
    def _orthonormal_beta_constraint_jacobian(self, coeffs, *beta_i):
        beta_i = np.reshape(np.array(beta_i), (self.y_dim, 1));
        return np.hstack((np.tile(0, (1, self.x_dim + self.x_dim)), np.matmul(beta_i.T, self.cov_y.T)));
    
    def _binary_alpha_constraint(self, coeffs, *args):
        alpha_index = args[0];
        alpha_pos = coeffs[alpha_index];
        alpha_neg = coeffs[alpha_index + self.x_dim];
        return alpha_pos * alpha_neg;
    
    def _binary_alpha_constraint_jacobian(self, coeffs, *args):
        alpha_index = args[0];
        alpha_pos = coeffs[alpha_index];
        alpha_neg = coeffs[alpha_index + self.x_dim];
        grad = np.zeros((1, self.x_dim + self.x_dim + self.y_dim));
        grad[0][alpha_index] = alpha_neg;
        grad[0][alpha_index + self.x_dim] = alpha_pos;
        return grad;
    
    # Execute method to return canonical correlation coefficients
    def get_canonical_correlations(self, **kwargs):
        
        # Minimum of x and y dimension to get number of canonical correlations
        min_x_y = min(self.x_dim, self.y_dim);
        
        # Or get number of directions as specified by user
        if "no_of_directions" in kwargs.keys():
            no_of_directions = kwargs["no_of_directions"];
        else:
            no_of_directions = min_x_y;
        
        # Evaluate minimum between the 2 values
        n = min(min_x_y, no_of_directions);
        
        # Define an array of all constraints
        constraints = np.array([]).astype(object);
        if self.constraints != None:
            constraints = np.append(constraints, self.constraints);
        for i in range(self.x_dim):
            constraints = np.append(constraints, {"type": "eq", "fun": self._binary_alpha_constraint, 
                                                  "jac": self._binary_alpha_constraint_jacobian, "args": [i]});
        constraints = np.append(constraints, {"type": "eq", "fun": self._norm_alpha_var, "jac": self._norm_alpha_var_constraint_jacobian});
        constraints = np.append(constraints, {"type": "eq", "fun": self._norm_beta_var, "jac": self._norm_beta_var_constraint_jacobian});
        
        # Initialize an array containing values of alpha vector for all canonical correlations
        self.alpha = np.zeros((n, self.x_dim));
        
        # Initialize an array containing values of alpha vector for all canonical correlations
        self.beta = np.zeros((n, self.y_dim));
        
        # Initialize an array containing values of all canonical correlations
        self.correlations = np.zeros(n);
        
        for i in range(n):
            temp_constraints = np.array(constraints);
            for j in range(i):
                temp_constraints = np.append(temp_constraints, {"type": "eq", "fun": self._orthonormal_alpha_constraint, 
                                                                "jac": self._orthonormal_alpha_constraint_jacobian, "args": self.alpha[j]});
                temp_constraints = np.append(temp_constraints, {"type": "eq", "fun": self._orthonormal_beta_constraint, 
                                                                "jac": self._orthonormal_beta_constraint_jacobian, "args": self.beta[j]});
             
            # Initialize array of alpha and beta coefficients, and optimized correlation
            temp_coeffs = np.array([]);
            max_correlation = 0;
            
            # Initialize parameters for generating random starting points for optimization
            init_points = 25;
            init_random_mean = 0;
            init_random_std = 7.5;
            optimization_output_coeffs = np.zeros(init_points).astype(object);
            optimization_output_correlations = np.zeros(init_points);
            
            success_counter = 0;
            for k in range(init_points):
                output = minimize(fun=self._objective_function, jac=self._objective_jacobian, constraints=temp_constraints, 
                                  bounds=np.vstack((np.tile([0, None], (self.x_dim + self.x_dim, 1)), np.tile([None, None], (self.y_dim, 1)))),
                                  x0=init_random_mean + (init_random_std * np.vstack((abs(np.random.randn(self.x_dim + self.x_dim, 1)), np.random.randn(self.y_dim, 1)))),
                                  method="SLSQP", options={"maxiter": 10000});
                
                # If the optimization converges, then store the value, else store 0 as dummy value
                if output["success"]:
                    success_counter += 1;
                    optimization_output_coeffs[k] = [round(coeff, 3) for coeff in list(output["x"])];
                    optimization_output_correlations[k] = -round(float(output["fun"]), 3);
                else:
                    optimization_output_coeffs[k] = [0];
                    
            # If none of the optimizations converge, return error
            if success_counter == 0:
                print("Numerical Optimization did not converge for any of the starting points");
                sys.exit();
                
            # Else return the set of coefficients corresponding to the maximum correlation
            else:
                max_correlation_idx = np.argsort(optimization_output_correlations)[::-1][0];
                max_correlation = optimization_output_correlations[max_correlation_idx];
                temp_coeffs = optimization_output_coeffs[max_correlation_idx];
                alpha = np.zeros(self.x_dim);
                for l in range(self.x_dim):
                    alpha[l] = temp_coeffs[l] - temp_coeffs[l + self.x_dim];
                
            self.alpha[i], self.beta[i] = alpha, temp_coeffs[-self.y_dim:];   
            self.correlations[i] = max_correlation;
            
        return self.alpha, self.beta;
    