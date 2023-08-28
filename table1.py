# Comparative Table of Unregulated and Regulated Indemnity Scenarios 
# (c) 2023 Yann Braouezec and John Cagnol
# Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, refer to http://creativecommons.org/licenses/by-nc/4.0/

import math
import numpy as np
import onecontract as oc


# Parameters for the type of problem
w = 1.0
l = 1.0

# Beta distributon (element of the EZ-square)
E = 0.05
Z = 0.989
a,b = oc.EZ2ab(E,Z)
fparam = ['Beta',a,b]

# Internal numerical parameter
tol = 1e-3


def growth(v1,v2):

    if abs(v1)>tol:
        increase = 100*(v2-v1)/v1
    else:
        increase = np.inf
            
    return(increase)



print("rho\tInd.\tTake up\tProfit\tPrice")

for rho in [0.5, 1, 2.5, 5, 7.5, 10]:

    Uparam = ['CARA',rho]
            
    t_unregulated, r_unregulated = oc.argmax_PI(l,w,Uparam,fparam)

    if abs(t_unregulated-1)<tol or abs(r_unregulated)<tol:
        
        r_unregulated = 0
        t_unregulated = 1
        p_unregulated = 0
        profit_unregulated = 0
        coverage_unregulated = 0
        SW_unregulated = 0
        epsilon_unregulated = -np.inf
        MC_unregulated = oc.p_bar(t_unregulated,r_unregulated,l,w,Uparam)
                
    else:
                
        p_unregulated = oc.p_bar(t_unregulated,r_unregulated,l,w,Uparam)
        profit_unregulated = oc.PI([t_unregulated,r_unregulated],l,w,Uparam,fparam)
        coverage_unregulated = 1-oc.F(t_unregulated,fparam)
        SW_unregulated = oc.SW(t_unregulated,1.0,r_unregulated,l,w,Uparam,fparam)
        epsilon_unregulated = -(oc.f(t_unregulated,fparam)*p_unregulated)/(oc.dp_bar_dtheta(t_unregulated,r_unregulated,l,w,Uparam)*(1-oc.F(t_unregulated,fparam)))
        MC_unregulated = p_unregulated*(1+1/epsilon_unregulated)


    # Now we look the the best SW in the regulated case
            
    Nr = 200
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
        
        t_regulated = oc.theta_star(r_regulated,l,w,Uparam,fparam)

        if abs(t_regulated-1)<tol or abs(r_regulated)<tol:
        
            r_regulated = 0
            t_regulated = 1
            p_regulated = 0
            profit_regulated = 0
            coverage_regulated = 0
            SW_regulated = 0
            epsilon_regulated = -np.inf
            MC_regulated = oc.p_bar(t_regulated,r_regulated,l,w,Uparam)
            
        else:
                
            p_regulated = oc.p_star(r_regulated,l,w,Uparam,fparam)
            profit_regulated = oc.PI([t_regulated,r_regulated],l,w,Uparam,fparam)
            coverage_regulated = 1-oc.F(t_regulated,fparam)
            SW_regulated = oc.SW(t_regulated,1.0,r_regulated,l,w,Uparam,fparam)
            epsilon_regulated = -(oc.f(t_regulated,fparam)*p_regulated)/(oc.dp_bar_dtheta(t_regulated,r_regulated,l,w,Uparam)*(1-oc.F(t_regulated,fparam)))
            MC_regulated = p_regulated*(1+1/epsilon_regulated)


    r_growth = growth(r_unregulated,r_regulated)
    p_growth = growth(p_unregulated,p_regulated)
    cov_growth = growth(1-oc.F(t_unregulated,fparam),1-oc.F(t_regulated,fparam))
    profit_growth = growth(profit_unregulated,profit_regulated)


    print("%.1f\t%+.1f%%\t%+.1f%%\t%+.1f%%\t%+.1f%%"%(rho,r_growth,cov_growth,profit_growth,p_growth))
