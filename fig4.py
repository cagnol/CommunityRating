# Analyzing Insurance Contracts Along the Vertical Section of the EZ-Square
# (c) 2023 Yann Braouezec and John Cagnol
# Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, refer to http://creativecommons.org/licenses/by-nc/4.0/

import math
import numpy as np
import onecontract as oc
import matplotlib.pyplot as plt



# Parameters for the type of problem
w = 1.0
l = 1.0
r = 1.0

# Utility function
Uparam = ['CARA',5.0]

# Section of the EZ-square to analyze
E = 0.05

# Internal numerical parameter
tol = 1e-3

# Discretization of the section
NZ = 1999
hZ= 1.0/(NZ+1)



Z_tab = np.arange(NZ,dtype=float)
ps_tab = np.arange(NZ,dtype=float)
ts_tab  = np.arange(NZ,dtype=float)
A_tab  = np.arange(NZ,dtype=float)
profit_tab = np.arange(NZ,dtype=float)
coverage_tab = np.arange(NZ,dtype=float)
SW_tab = np.arange(NZ,dtype=float)
epsilon_tab = np.arange(NZ,dtype=float)
MC_tab = np.arange(NZ,dtype=float)
var_tab = np.arange(NZ,dtype=float)

i=0

for Z in np.linspace(0.001, 0.998, NZ):

    a,b = oc.EZ2ab(E,Z)
    fparam = ['Beta',a,b]
    
    Z_tab[i] = Z
    
    ps = oc.p_star(r,l,w,Uparam,fparam)
    ts = oc.theta_star(r,l,w,Uparam,fparam)
    
    if abs(ts-1)<tol:
        
        ts = 1
        ts_tab[i] = ts
        ps_tab[i] = 0
        profit_tab[i] = 0
        coverage_tab[i] = 0
        SW_tab[i] = 0
        epsilon_tab[i] = -(oc.f(ts,fparam)*oc.p_bar(ts,r,l,w,Uparam))/(oc.dp_bar_dtheta(ts,r,l,w,Uparam)*(1-oc.F(ts,fparam)))
        MC_tab[i] = oc.p_bar(ts,r,l,w,Uparam)
        A_tab[i] = 1
        
    else :
        
        ts_tab[i] = ts
        ps_tab[i] = ps
        profit_tab[i] = oc.PI([ts,r],l,w,Uparam,fparam)
        coverage_tab[i] = 1-oc.F(ts,fparam)
        SW_tab[i] = oc.SW(ts,1.0,r,l,w,Uparam,fparam)
        epsilon_tab[i] = -(oc.f(ts,fparam)*oc.p_bar(ts,r,l,w,Uparam))/(oc.dp_bar_dtheta(ts,r,l,w,Uparam)*(1-oc.F(ts,fparam)))
        MC_tab[i] = oc.p_bar(ts,r,l,w,Uparam)*(1+1/epsilon_tab[i])
        A_tab[i] = oc.A(ts,fparam)
            
    i=i+1

plt.figure(figsize=(10,15))
            
plt.subplot(3, 1, 1)
plt.plot(Z_tab, profit_tab, 'r', label='profit')
plt.plot(Z_tab, SW_tab, 'y', label='social welfare')             
plt.xlabel('Z')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(Z_tab, coverage_tab, 'c', label='take up rate')
plt.plot(Z_tab, ts_tab, 'r', label='theta*')
plt.plot(Z_tab, A_tab, 'k', label='A(theta*)')
plt.xlabel('Z')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(Z_tab, ps_tab, 'b', label='p*')
plt.plot(Z_tab, MC_tab, 'pink', label='marginal cost')
plt.xlabel('Z')
plt.legend()

plt.savefig('fig4.png')
plt.clf()
