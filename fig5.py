# Comparative Analysis of Unregulated and Regulated Indemnity Scenarios 
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

# Beta distributon (element of the EZ-square)
E = 0.05
Z = 0.989

# Discretization of the section
NZ = 1999
hZ= 1.0/(NZ+1)

# x-axis and its discretization
rho_min = 0.5
rho_max = 50.0
Nrho = 400 
hrho = 1.0/(Nrho+1)

# Disctretization of [0,l] to look for the best r in the regulated case
rmin = 0.001
rmax = 0.999
Nr = 200
hr = 1.0/(Nr+1)

# Internal numerical parameter
tol = 1e-3



# Define the relative grown between tab1 and tab2

def growth(tab1,tab2):

    size = len(tab1)
    assert(len(tab2)==size)
    
    increase = np.arange(size,dtype=float)

    for i in range(size):
        if abs(tab1[i])>tol:
            increase[i] = 100*(tab2[i]-tab1[i])/tab1[i]
        else:
            increase[i] = np.inf

    return(increase)



# Creation of the Beta distribution

a,b = oc.EZ2ab(E,Z)
fparam = ['Beta',a,b]


# Creation of all of tables

rho_tab = np.arange(Nrho,dtype=float)

p_unregulated_tab = np.arange(Nrho,dtype=float)
t_unregulated_tab  = np.arange(Nrho,dtype=float)
r_unregulated_tab  = np.arange(Nrho,dtype=float)
A_unregulated_tab  = np.arange(Nrho,dtype=float)
profit_unregulated_tab = np.arange(Nrho,dtype=float)
coverage_unregulated_tab = np.arange(Nrho,dtype=float)
SW_unregulated_tab = np.arange(Nrho,dtype=float)
epsilon_unregulated_tab = np.arange(Nrho,dtype=float)
MC_unregulated_tab = np.arange(Nrho,dtype=float)
AC_unregulated_tab = np.arange(Nrho,dtype=float)

p_regulated_tab = np.arange(Nrho,dtype=float)
t_regulated_tab  = np.arange(Nrho,dtype=float)
r_regulated_tab  = np.arange(Nrho,dtype=float)
A_regulated_tab  = np.arange(Nrho,dtype=float)
profit_regulated_tab = np.arange(Nrho,dtype=float)
coverage_regulated_tab = np.arange(Nrho,dtype=float)
SW_regulated_tab = np.arange(Nrho,dtype=float)
epsilon_regulated_tab = np.arange(Nrho,dtype=float)
MC_regulated_tab = np.arange(Nrho,dtype=float)
AC_regulated_tab = np.arange(Nrho,dtype=float)


# Exploration of the various rho

i=0

for rho in np.linspace(rho_min, rho_max, Nrho):
    
    Uparam = ['CARA',rho]
    
    rho_tab[i] = rho


    # The scenario where the indemnity is not regulated
    
    t_unregulated, r_unregulated = oc.argmax_PI(l,w,Uparam,fparam)
    
    if abs(t_unregulated-1)<tol or abs(r_unregulated)<tol:
        
        r_unregulated = 0
        t_unregulated = 1
        p_unregulated = 0
        profit_unregulated = 0
        coverage_unregulated = 0
        SW_unregulated = 0
        epsilon_unregulated = -np.inf
        A_unregulated = 1
        MC_unregulated = oc.p_bar(t_unregulated,r_unregulated,l,w,Uparam)
                
    else:
                
        p_unregulated = oc.p_bar(t_unregulated,r_unregulated,l,w,Uparam)
        profit_unregulated = oc.PI([t_unregulated,r_unregulated],l,w,Uparam,fparam)
        coverage_unregulated = 1-oc.F(t_unregulated,fparam)
        SW_unregulated = oc.SW(t_unregulated,1.0,r_unregulated,l,w,Uparam,fparam)
        epsilon_unregulated = -(oc.f(t_unregulated,fparam)*p_unregulated)/(oc.dp_bar_dtheta(t_unregulated,r_unregulated,l,w,Uparam)*(1-oc.F(t_unregulated,fparam)))
        MC_unregulated = p_unregulated*(1+1/epsilon_unregulated)
        A_unregulated = oc.A(t_unregulated,fparam)
        
    t_unregulated_tab[i] = t_unregulated
    r_unregulated_tab[i] = r_unregulated
    p_unregulated_tab[i] = p_unregulated
    profit_unregulated_tab[i] = profit_unregulated
    coverage_unregulated_tab[i] = coverage_unregulated
    SW_unregulated_tab[i] = SW_unregulated
    MC_unregulated_tab[i] = MC_unregulated
    epsilon_unregulated_tab[i] = epsilon_unregulated             
    A_unregulated_tab[i] = A_unregulated
    AC_unregulated_tab[i] = A_unregulated_tab[i] * r_unregulated


    # The scenario where the indemnity is regulated
    # Wwe look the the best SW in the regulated case
            
    SW_tab_r = np.arange(Nr,dtype=float)
    
    j=0
    
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
        A_regulated = 1
        
    else:
                
        p_regulated = oc.p_star(r_regulated,l,w,Uparam,fparam)
        profit_regulated = oc.PI([t_regulated,r_regulated],l,w,Uparam,fparam)
        coverage_regulated = 1-oc.F(t_regulated,fparam)
        SW_regulated = oc.SW(t_regulated,1.0,r_regulated,l,w,Uparam,fparam)
        epsilon_regulated = -(oc.f(t_regulated,fparam)*p_regulated)/(oc.dp_bar_dtheta(t_regulated,r_regulated,l,w,Uparam)*(1-oc.F(t_regulated,fparam)))
        MC_regulated = p_regulated*(1+1/epsilon_regulated)
        A_regulated = oc.A(t_regulated,fparam)
        
    t_regulated_tab[i] = t_regulated
    r_regulated_tab[i] = r_regulated
    p_regulated_tab[i] = p_regulated
    profit_regulated_tab[i] = profit_regulated
    coverage_regulated_tab[i] = coverage_regulated
    SW_regulated_tab[i] = SW_regulated
    MC_regulated_tab[i] = MC_regulated
    epsilon_regulated_tab[i] = epsilon_regulated             
    A_regulated_tab[i] = A_regulated
    AC_regulated_tab[i] = A_regulated_tab[i] * r_regulated
    
    i=i+1


# Plot the results
        
plt.figure(figsize=(30,30))
            
plt.subplot(3, 3, 1)
plt.plot(rho_tab, profit_unregulated_tab, 'r', label='profit')
plt.plot(rho_tab, SW_unregulated_tab, 'y', label='social welfare')             
plt.xlabel('rho')
plt.legend()
plt.title('CARA  E=%.2f  Z=%.3f (var=%.3e, CV=%.4f)  l=%.2f\nUNREGULATED INDEMNITY: r is endogeneous'%(E,Z,oc.variance(fparam),oc.CV(fparam),l),fontsize = 8)

plt.subplot(3, 3, 4)
plt.plot(rho_tab, coverage_unregulated_tab, 'c', label='take up rate')
plt.plot(rho_tab, t_unregulated_tab, 'r', label='theta*')
plt.plot(rho_tab, A_unregulated_tab, 'k', label='A(theta*)')
plt.plot(rho_tab, r_unregulated_tab, 'g', label='r')
plt.xlabel('rho')
plt.legend()

plt.subplot(3, 3, 7)
plt.plot(rho_tab, p_unregulated_tab, 'b', label='p*')
plt.plot(rho_tab, MC_unregulated_tab, 'pink', label='marginal cost')
plt.plot(rho_tab, AC_unregulated_tab, 'olive', label='average cost')
plt.xlabel('rho')
plt.legend()


plt.subplot(3, 3, 2)
plt.plot(rho_tab, profit_regulated_tab, 'r', label='profit')
plt.plot(rho_tab, SW_regulated_tab, 'y', label='social welfare')             
plt.xlabel('rho')
plt.legend()
plt.title('CARA  E=%.2f  Z=%.3f (var=%.3e, CV=%.4f)  l=%.2f \nREGULATED INDEMNITY: r is chosen to maximize SW'%(E,Z,oc.variance(fparam),oc.CV(fparam),l),fontsize = 8)

plt.subplot(3, 3, 5)
plt.plot(rho_tab, coverage_regulated_tab, 'c', label='take up rate')
plt.plot(rho_tab, t_regulated_tab, 'r', label='theta*')
plt.plot(rho_tab, A_regulated_tab, 'k', label='A(theta*)')
plt.plot(rho_tab, r_regulated_tab, 'g', label='R')
plt.xlabel('rho')
plt.legend()

plt.subplot(3, 3, 8)
plt.plot(rho_tab, p_regulated_tab, 'b', label='p*')
plt.plot(rho_tab, MC_regulated_tab, 'pink', label='marginal cost')
plt.plot(rho_tab, AC_regulated_tab, 'olive', label='average cost')
plt.xlabel('rho')
plt.legend()


plt.subplot(3, 3, 3)
plt.plot(rho_tab, growth(profit_unregulated_tab,profit_regulated_tab), 'r', label='profit')
plt.plot(rho_tab, growth(SW_unregulated_tab,SW_regulated_tab), 'y', label='social welfare')             
plt.xlabel('rho')
plt.legend()
plt.title('CARA  E=%.2f  Z=%.3f  (var=%.3e, CV=%.4f)  l=%.2f\nPercent increase from UNREGULATED to REGULATED'%(E,Z,oc.variance(fparam),oc.CV(fparam),l),fontsize = 8)

plt.subplot(3, 3, 6)
plt.plot(rho_tab, growth(coverage_unregulated_tab,coverage_regulated_tab), 'c', label='take up rate')
plt.plot(rho_tab, growth(t_unregulated_tab,t_regulated_tab), 'r', label='theta*')
plt.plot(rho_tab, growth(A_unregulated_tab,A_regulated_tab), 'k', label='A(theta*)')
plt.plot(rho_tab, growth(r_unregulated_tab,r_regulated_tab), 'g', label='R')
plt.xlabel('rho')
plt.legend()
            
plt.subplot(3, 3, 9)
plt.plot(rho_tab, growth(p_unregulated_tab,p_regulated_tab), 'b', label='p*')
plt.plot(rho_tab, growth(MC_unregulated_tab,MC_regulated_tab), 'pink', label='marginal cost')
plt.plot(rho_tab, growth(AC_unregulated_tab,AC_regulated_tab), 'olive', label='average cost')
plt.xlabel('rho')
plt.legend()

plt.savefig('fig5.png')
