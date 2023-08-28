# Exploration EZ-Square - Relative Increase of the Indemnity in Regulated vs. Unregulated Scenarios
# (c) 2023 Yann Braouezec and John Cagnol
# Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, refer to http://creativecommons.org/licenses/by-nc/4.0/


import math
import numpy as np
import onecontract as oc
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools



# Parameters for the type of problem
w = 1.0
l = 1.0

# Utility function
Uparam = ['CARA',5]

# Internal numerical parameter
tol = 1e-3

# Discretization of the EZ-square
h=0.025




def growth(v1,v2):

    if abs(v1)>tol:
        increase = 100*(v2-v1)/v1
    else:
        increase = np.inf
            
    return(increase)



@np.vectorize
def Compute_r_growth(e,z):
    a,b = oc.EZ2ab(e,z)
    fparam = ['Beta',a,b]

    t_unregulated, r_unregulated = oc.argmax_PI(l,w,Uparam,fparam)

    # Now we look the the best SW in the regulated case
            
    Nr = 100
    hr = 1.0/(Nr+1)

    SW_tab_r = np.arange(Nr,dtype=float)

    j=0

    rmin = 0.001
    rmax = 0.999
    
    for r in np.linspace(rmin, rmax, Nr):
        
        ps = oc.p_star(r,l,w,Uparam,fparam)
        ts = oc.theta_star(r,l,w,Uparam,fparam)

        if abs(ts-1)<tol:
            SW_tab_r[j] = 0
        else :
            SW_tab_r[j] = oc.SW(ts,1.0,r,l,w,Uparam,fparam)

        j=j+1

    j_swopt = np.argmax(SW_tab_r)
    r_regulated = rmin + j_swopt * hr

    if abs(r_regulated)<tol:        
        r_regulated = 0            

    if abs(r_regulated)>tol:
        return growth(r_unregulated,r_regulated)
    else:
        return np.inf



E_tab, Z_tab = np.meshgrid(
    np.arange(h/10,1-h/10,h,dtype=float),
    np.arange(h/10,1-h/10,h,dtype=float))


levels = np.linspace(0, 55, 56)
    
rgrowth_tab = Compute_r_growth(E_tab,Z_tab)

plt.contourf(E_tab,Z_tab,rgrowth_tab, levels, cmap='rainbow')
plt.colorbar();
plt.title('Relative Increase of the Indemnity')
plt.xlabel('E')
plt.ylabel('Z')
plt.savefig('fig6.png')
plt.clf()
    
