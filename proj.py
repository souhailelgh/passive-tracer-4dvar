# coding: utf-8
# written by SOUHAIL ELGHAYAM on November 21, 2020

import numpy as np
import math
from numpy.linalg import inv


# strong constraint 4D-var
#=========================
print("Strong constraint : \n===============\n\n")

# définition des variables 

sig2_obs =  0.001* 0.001   # variance d'erreur d'obs
sig2_q = 1*1         # variance d'erreur de q
sig2_b = sig2_q      # variance d'erreur du biais
beta = 0.5              # le biais 
u = 1                # le vent  
print('\nthe wind u : ',np.round(u,4),
      '\nbaias beta : ',np.round(beta,4))

xt = np.array([[2.5, 3.4, 1.3, beta]]).transpose()  # vecteur de l'état réelle à t=0
print('\ntruth vector at t0 :\n',np.round(xt,4))

xb_0 = np.array([[2, 3, 1, 0]]).transpose()  # l'ébauche à t=0
print('\nBackgroung vector at t0 :\n',np.round(xb_0,4))


R = np.array([[sig2_obs, 0, 0 ],
              [0, sig2_obs, 0 ],
              [0, 0, sig2_obs ]]) # matrice de covariance des erreurs d'obs
print('\nObsevation error variance matrix :\n',np.round(R,4))

B = np.array([[sig2_q, 0, 0, 0 ],
              [0, sig2_q, 0, 0 ],
              [0, 0, sig2_q, 0 ],
              [0, 0, 0, sig2_b ]]) # matrice de covariance des erreurs d'ébauche
print('\nBackground error variance matrix :\n',np.round(B,4))

H = np.array([[1, 0, 0, 1],
              [0, 1, 0, 1],
              [0, 0, 1, 1]]) # Operateur d'obs
print('\nobs operator :\n',np.round(H,4))    

M = np.array([[ 1    , -u/2 , +u/2, 0],
              [ +u/2 , 1    , -u/2, 0],
              [ -u/2 , +u/2 , 1   , 0],
              [ 0    , 0    , 0   , 1]]) # matrice du modèle imparfait
print('\nimperfect model : \n',np.round(M,4)) 

# L = np.array([[ 1    , -u/2 , +u/2 ],
#               [ +u/2 , 1    , -u/2 ],
#               [ -u/2 , +u/2 , 1    ]]) # matrice du modèle linéaire tangent
# print('\ntangent linear model : \n',np.round(L,4))

y0=np.dot(H, M.dot(xt))  # obserrvation a t1
print('\nObservation at t1 : \n',np.round(y0,4))

d =y0-np.dot(np.dot(H,M),xb_0) # inovation a t1
print('\nInnovation at t1 : \n',np.round(d,4))

#calcul de l'analys à t0
HM=np.dot(H,M)
HMT=np.transpose(HM)
BHMT=np.dot(B,HMT)
HMB=np.dot(HM,B)
HMBHMT=np.dot(HMB,HMT)
K_0=np.dot(BHMT,inv(np.matrix(HMBHMT+R)))
xa_0= xb_0 + np.dot(K_0,d)
print('\nAnalysis at t0 : \n',np.round(xa_0,4))

#calcul de l'analys à t1
HT=np.transpose(H)
BHT=np.dot(B,HT)
HB=np.dot(H,B)
HBHT=np.dot(HB,HT)
K_1=np.dot(BHT,inv(np.matrix(HBHT+R)))
xb_1 = np.dot(M,xb_0)
xa_1= xb_1 + np.dot(K_1,d)
print('\nAnalysis at t1 : \n',np.round(xa_1,4))


# weak constraint 4D-var
#=========================
print("\nWeak constraint : \n===============\n\n")

# définition des variables 

sig2_obs =  0.001* 0.001   # variance d'erreur d'obs
sig2_q = 1*1         # variance d'erreur de q
sig2_b = sig2_q      # variance d'erreur du biais
sig2_m = sig2_q      # variance d'erreur du modèle
beta = 0.5           # le biais 
u = 1                # le vent 
k = 0.4              # le vent  

print('\ndiffusion coef :', np.round(k,4),
      '\nthe wind u : ',np.round(u,4),
      '\nbaias beta : ',np.round(beta,4))

xt = np.array([[2.5, 3.4, 1.3, beta]]).transpose()  # vecteur de l'état réelle à t=0
print('\ntruth vector at t0 :\n',np.round(xt,4))

xb_0 = np.array([[2, 3, 1, 0]]).transpose()  # l'ébauche à t=0
print('\nBackgroung vector at t0 :\n',np.round(xb_0,4))

Q = np.array([[sig2_m, 0, 0, 0],
              [0, sig2_m, 0, 0],
              [0, 0, sig2_m, 0],
              [0, 0, 0     , 0]]) # matrice de covariance des erreurs du modèle
print('\nmodel error variance matrix :\n',np.round(Q,4))

K=k*1.5/math.pi
Mt = np.array([[ 1-2*K , K-u/2 , K+u/2, 0],
               [ K+u/2 , 1-2*K , K-u/2, 0],
               [ K-u/2 , K+u/2 , 1-2*K, 0],
               [ 0     , 0     , 0    , 1]]) # matrice du  modèle vrai
print('\ntrue model :\n',np.round(Mt,4)) 

y0=np.dot(H, Mt.dot(xt))  # obserrvation a t1
print('\nObservation at t1 : \n',np.round(y0,4))

d =y0-np.dot(np.dot(H,M),xb_0) # inovation a t1
print('\nInnovation at t1 : \n',np.round(d,4))

#calcul de l'analys à t0

HT=H.transpose()
MT=M.transpose()
MB   = np.dot(M,B)
MBMT = np.dot(MB,MT)
X    = MBMT +Q
Y    = np.dot(H,X.dot(HT))
BMT  = np.dot(B,MT)
BMTHT= np.dot(BMT,HT)
delta_0= np.dot(BMTHT,inv(Y+R)).dot(d)

xa_0 = xb_0 + delta_0
print('\nAnalysis vector at t0 :\n',np.round(xa_0,4))

#calcul de l'analys à t1

delta_1= np.dot(X.dot(HT),inv(Y+R)).dot(d)
xb_1   = M.dot(xb_0)

xa_1 = xb_1 + delta_1
print('\nAnalysis vector at t1 :\n',np.round(xa_1,4))

#the end.
