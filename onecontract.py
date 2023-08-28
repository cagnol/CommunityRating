# Main functions to simulate the behavior of a private monopolist insurer within the context of community rating.
# (c) 2023 Yann Braouezec and John Cagnol
# Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, refer to http://creativecommons.org/licenses/by-nc/4.0/


import math
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import newton
from scipy.optimize import root_scalar
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from scipy.misc import derivative
from scipy import integrate
from scipy.stats import beta
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import warnings

# Function used to warn of a problem during runtime

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s\n' % (category.__name__, message)
warnings.formatwarning = warning_on_one_line

# If FASTEXEC is set to True, the optimization algorithm will be faster but less accurate.

FASTEXEC = False



####################
# Utility function #
####################

# Define non-dimensional function U (denoted u in the article)

def U(w,Uparam):
   if Uparam[0]=='CARA':
      rho = Uparam[1]
      return (1-math.exp(-rho*w))/(1-math.exp(-rho))
   else:
      raise ValueError('unknown utility function type')

# Print the utiliy function

def U_print(Uparam):
   if Uparam[0]=='CARA':
      rho = Uparam[1]
      return "CARA(%.2f)"%rho
   elif Uparam[0]=='sqrt':
      return "sqrt"
   else:
      raise ValueError('unknown utility function type')
    

############################
# Agent Types (measure mu) #
############################

# Mapping from and to the EZ-square

def EZ2ab (E,Z):
   assert (E>0 and E<1)
   assert (Z>0 and Z<1)
   return E*Z/((1-E)*(1-Z)), Z/(1-Z)


def ab2EZ (a,b):
   assert (a>0 and b>0)
   return a/(a+b), b/(b+1)

# PDF

def f(theta,fparam):
   if fparam[0]=='Beta':
      a = fparam[1]
      b = fparam[2]
      return beta.pdf(theta,a,b)
   else:
      raise Exception('unknown density function type')
        
def f_print(fparam):
   if fparam[0]=='Beta':
      a = fparam[1]
      b = fparam[2]
      return "Beta(%.2f,%.2f)"%(a,b)
   else:
      raise Exception('unknown density function type')
    
# CDF

def F(theta,fparam):
   if fparam[0]=='Beta':
      a = fparam[1]
      b = fparam[2]
      return beta.cdf(theta,a,b)
   else:
      res, err = integrate.quad(f, 0, theta, args=(a,b))
      return res

# Expectancy

def xf(theta,fparam):
   return theta*f(theta,fparam)

def expectancy(fparam):
   if fparam[0]=='Beta':
      a = fparam[1]
      b = fparam[2]
   return a/(a+b)

# Variance

def x2f(theta,fparam):
   return theta*theta*f(theta,fparam)

def variance(fparam):
   if fparam[0]=='Beta':
      a = fparam[1]
      b = fparam[2]
   return a*b/(((a+b)**2)*(a+b+1))

# Coeffcient of variation

def CV(fparam):
   return(math.sqrt(variance(fparam))/expectancy(fparam))

def CoefVariation(theta,fparam):
   tol = 1e-4 # when theta is close to 1, the coefficient of variation is 0

   if theta<0 or theta>1:
      warnings.warn("CoefVariation called with theta=%e for f=%s"%(theta,f_print(fparam)), RuntimeWarning)

   denominator = 1.0-F(theta,fparam)

   if abs(theta-1)<tol:
      return 0
   else:
      res, err =integrate.quad(x2f, theta, 1.0, args=(fparam))
      Atheta = A(theta,fparam)
      Vtheta = res / denominator - Atheta**2
      return (math.sqrt(Vtheta)/Atheta)

# Average probability

def A(theta,fparam):
   tol = 1e-4    # when theta is close to 1, A(theta,fparam) is replaced by the limit in 1 (which is 1)
   tol2 = 1e-8   # when the denominator is too close to 0, a Taylor expansion is performed.

   if theta<0 or theta>1:
      warnings.warn("A called with theta=%e for f=%s"%(theta,f_print(fparam)), RuntimeWarning)

   denominator = 1.0-F(theta,fparam)
      
   if abs(theta-1)<tol:
      return 1
   elif abs(denominator)<tol2 and fparam[0]=='Beta':
      b = fparam[2]
      return 1 - (1-theta)*b/(b+1)   
   else: 
      res, err =integrate.quad(xf, theta, 1.0, args=(fparam))
      return res / denominator

# Numerically test of the convexity/concavity of the average probability.
# Returns two flags.

def A_convexity(fparam):

   N = 20
   h = 1/(N-1)
      
   y = np.arange(N,dtype=float)
   Atab = np.arange(N,dtype=float)

   i=0
   for y in np.linspace(0.0, 1, N):
      Atab[i] = A(y,fparam) 
      i=i+1

   is_concave = True
   is_convex = True

   i=0
   for y in np.linspace(0.0, 1, N):

      if i>1 and i<N-1:
         Apph2 = Atab[i+1]+Atab[i-1]-2*Atab[i]
         if Apph2<0:
            is_convex = False
         if Apph2>0:
            is_concave = False
               
      i=i+1

   return is_convex, is_concave
   

#####################
# ADVERSE SELECTION #
#####################

# Define V (see article)

def V(theta,p,r,l,w,Uparam):
    return theta*U(w-l+r-p,Uparam)+(1-theta)*U(w-p,Uparam)

# Define theta_bar

def theta_bar(p,r,l,w,Uparam):

   if Uparam[0]=='CARA':
      rho = Uparam[1]
      return (1-math.exp(-rho*p))/(1-math.exp(-rho*p)-math.exp(rho*(l-r))+math.exp(rho*(l-p)))
   else:
      return (U(w,Uparam)-U(w-p,Uparam))/(U(w,Uparam)-U(w-p,Uparam)+U(w-l+r-p,Uparam)-U(w-l,Uparam))  

# Define p_bar and its derivative with respect to theta

def p_equation(x,theta,r,l,w,Uparam):
    return V(theta,x,r,l,w,Uparam)-V(theta,0,0,l,w,Uparam)

def p_bar(theta,r,l,w,Uparam):
   tol = 1e-4
    
   if Uparam[0]=='CARA' and math.fabs(w-1)<tol:
        rho = Uparam[1]
        return(1 / rho * math.log((theta * math.exp(rho * l) + 1 - theta) / (theta * math.exp(rho * (l - r)) + 1 - theta)))
   else:
        x0 = .8
        res = fsolve(p_equation,x0,args=(theta,r,l,w,Uparam))
        return res[0]

def dp_bar_dtheta (theta,r,l,w,Uparam):  
   tol = 1e-4
        
   if Uparam[0]=='CARA' and math.fabs(w-1)<tol:
        rho = Uparam[1]
        return (math.exp(rho * l) - math.exp(rho * (l - r))) / (theta * math.exp(rho * l) + 1 - theta) / (theta * math.exp(rho * (l - r)) + 1 - theta) / rho
   else:
        h = 0.001
        return derivative(p_bar, theta, dx=h, args=(r,l,w,Uparam))



###############
# OPTMIZATION #
###############

# Define the profit

def PI(X,l,w,Uparam,fparam):
   theta = X[0]
   r = X[1]
   return (p_bar(theta,r,l,w,Uparam)-r*A(theta,fparam)) * (1-F(theta,fparam))

# Compute (theta*,r*)
 
def minus_PI(X,l,w,Uparam,fparam):
   return -PI(X,l,w,Uparam,fparam)

def argmax_PI(l,w,Uparam,fparam):

        if FASTEXEC:
                res = minimize(minus_PI, (0.5,0.5), args=(l,w,Uparam,fparam), bounds=((0,1),(0,l)))
        else:
                res = dual_annealing(minus_PI, args=(l,w,Uparam,fparam), bounds=[(0,1),(0,l)])
                
        if res.success==False:
                warnings.warn("Optimization failed for U=%s and f=%s."%(U_print(Uparam),f_print(fparam)), RuntimeWarning)  

        return res.x

# Compute p* for a given r

def p_star_function_to_maximize(p,r,l,w,Uparam,fparam):
    tb = theta_bar(p,r,l,w,Uparam);
    if tb<1:
        return (p-r*A(tb,fparam))*(1-F(tb,fparam))
    else:
        return 0

def p_star_function_to_minimize(p,r,l,w,Uparam,fparam):
    return -p_star_function_to_maximize(p,r,l,w,Uparam,fparam)

def p_star(r,l,w,Uparam,fparam):
    res = minimize_scalar(p_star_function_to_minimize, bounds=(0,r), args=(r,l,w,Uparam,fparam), method='bounded')
    return res.x

# Compute theta* for a given r

def theta_star_function_to_maximize(theta,r,l,w,Uparam,fparam):
    pb = p_bar(theta,r,l,w,Uparam);
    if pb < r:
        return (pb-r*A(theta,fparam))*(1-F(theta,fparam))
    else:
        return 0

def theta_star_function_to_minimize(theta,r,l,w,Uparam,fparam):
    return -theta_star_function_to_maximize(theta,r,l,w,Uparam,fparam)

def theta_star(r,l,w,Uparam,fparam):
    res = minimize_scalar(theta_star_function_to_minimize, bounds=(0,1), args=(r,l,w,Uparam,fparam), method='bounded')
    return res.x
 
# Compute the Social Welfare SW

def v(theta,r,l,w,Uparam):
    return p_bar(theta,r,l,w,Uparam)-theta*r

def vf(theta,r,l,w,Uparam,fparam):
    return v(theta,r,l,w,Uparam)*f(theta,fparam)

def SW(theta_inf,theta_sup,r,l,w,Uparam,fparam):
    res, err = integrate.quad(vf, theta_inf, theta_sup, args=(r,l,w,Uparam,fparam)) 
    return res 
