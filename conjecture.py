# Investigating the Conjecture: Convexity of Function A for alpha > 1
# (c) 2023 Yann Braouezec and John Cagnol
# Licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, refer to http://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import onecontract as oc
import matplotlib.pyplot as plt
from matplotlib import cm


# The conjecture will be checked for (a,b) in [0,A] x [0,B]
# A and B and defined below

A = 20
B = 20

# Discretizaton steps :

ha = 1e-2
hb = 1e-2 

# Discretizaton of [0,1] :
# N is the number of points, h the step size

N = 50
h = 1/(N-1)   




# Argumnts: a (alpha) and b (beta), parameters of the Beta distrbution
# Return two flags: is_convex, is_concave

def A_convexity(a,b):

   fparam = ['Beta',a,b]

   # Discretizaton of [0,1]
   
   y = np.arange(N,dtype=float)
   Atab = np.arange(N,dtype=float)

   # Compute A on the each node of the discretization
   
   i=0
   for y in np.linspace(0.0, 1, N):
      Atab[i] = oc.A(y,fparam) 
      i=i+1

   # Check the convexity using second order finite differences
      
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



# Discretizing [0,A] x [0,B]   

a_tab = np.arange(ha/100,A-ha/100,h,dtype=float)
b_tab = np.arange(hb/100,B-hb/100,h,dtype=float)


# Checking on each point of the grid

Conjecture = True

for a in a_tab:
   for b in b_tab:
      convexity = [A_convexity(a,b)]
      Conjecture = Conjecture and convexity[0]
      if not convexity[0]:
         print(a,b)

if Conjecture:
   print("Conjecture verified on [0,%e]x[0,%e]"%(A,B))
   print("Discretization steps : %e and %e"%(ha,hb))
   print("Discretizaton of (0,1) : %e"%h)
else:
   print("Conjecture NOT verified")
