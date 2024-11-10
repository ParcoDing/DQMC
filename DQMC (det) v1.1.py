#!/usr/bin/env python
"""
  Wang-Landau algorithm for 2D Ising model
  K. Haule, Feb.28, 2010
"""
import numpy as np
import random
import copy
from tqdm import trange,tqdm
import matplotlib.pyplot as plt
import os
import time
import numba as nb
from scipy.linalg import expm

def compute_M_left_M_right(K,V_ls,l):
    I = np.eye(N)
    M_up_left = np.eye(N)
    M_dn_left = np.eye(N)
    for l_ in range(nt-1,l,-1):
        V = V_ls[l_]
        M_up_left = np.dot(M_up_left, expm(-K))
        M_up_left = np.dot(M_up_left, expm(-nu*V))
        M_dn_left = np.dot(M_dn_left, expm(-K))
        M_dn_left = np.dot(M_dn_left, expm(-(-nu*V)))
    M_up_right = np.eye(N)
    M_dn_right = np.eye(N)
    for l_ in range(l-1,-1,-1):
        V = V_ls[l_]
        M_up_right = np.dot(M_up_right, expm(-K))
        M_up_right = np.dot(M_up_right, expm(-nu*V))
        M_dn_right = np.dot(M_dn_right, expm(-K))
        M_dn_right = np.dot(M_dn_right, expm(-(-nu*V)))
    return M_up_left, M_up_right, M_dn_left, M_dn_right


def create_periodic_lattice_matrix(L):
    N = L * L
    A = np.zeros((N, N), dtype=int)
    
    for y in range(L):
        for x in range(L):
            current = x + y * L
            # 右邻居
            right = ((x + 1) % L) + y * L
            A[current, right] = 1
            A[right, current] = 1  # 无向图，矩阵对称
            
            # 上邻居
            up = x + ((y + 1) % L) * L
            A[current, up] = 1
            A[up, current] = 1  # 无向图，矩阵对称
    return A


# 代码的参考来源: Lecture Notes on Advances of Numerical Methods for Quantum Monte Carlo Simulations of the Hubbard Model, ZHAOJUN BAI, WENBIN CHEN, RICHARD SCALETAR, ICHTTARO YAMAZAKI
# 代码的参考来源: HOW TO WRITE A DETERMINANT QMC CODE
# 代码的参考来源: https://quantummc.xyz/dqmc/ 
# 采样结果参考：http://ziyangmeng.iphy.ac.cn/files/teaching/SummerSchoolSimpleDQMCnoteXYX201608.pdf

# 定义 Hubbard model:
# H = H_{K} + H_{\mu} + H_{V}
# H_{K} = -t \sum_{\langle i,j\rangle,\sigma} (c_{i\sigma}^{\dagger} c_{j\sigma} + c_{j\sigma}^{\dagger} c_{i\sigma}),
# H_{\mu} = -\mu \sum_{i} (n_{i\uparrow} + n_{i\downarrow})
# H_{V} = U \sum_{i} (n_{i\uparrow} - \frac{1}{2})(n_{i\downarrow} - \frac{1}{2})

U = 5
mu = 5   
t = 1
#U = 5
mu = mu - U/2
dt = 0.1
nt = 10
beta = nt*dt
T = 1/beta
L = 2
N = L**2


nu = np.arccosh(np.exp(U*dt/2))


K = create_periodic_lattice_matrix(L)
K = -t*K
for i in range(N):
    K[i,i] = mu + U/2
K = dt*K

HS_field = np.random.choice([-1,1],(N,nt))
V_ls = []
for l_ in range(0,nt):
    V = np.zeros((N,N))
    np.fill_diagonal(V, HS_field[:,l_])
    V_ls.append(V)
V_ls = np.array(V_ls)
I = np.eye(N)

G_up_ls = []
G_dn_ls = []
D_ls = []
for sweep in trange(1200):
    for mini_sweep in range(N*nt):
        
        i = np.random.randint(N)
        l = np.random.randint(nt)

        M_up_left, M_up_right, M_dn_left, M_dn_right = compute_M_left_M_right(K,V_ls,l)
        M_up = M_up_left @ expm(-K) @ expm(-nu*V_ls[l]) @ M_up_right
        M_dn = M_dn_left @ expm(-K) @ expm(-(-nu*V_ls[l])) @ M_dn_right

        det_M_up = np.linalg.det(I + M_up)
        det_M_dn = np.linalg.det(I + M_dn)
        M_before = det_M_up*det_M_dn

        HS_before = HS_field[i,l]
        HS_field[i,l] = -1*HS_field[i,l]
        V_ls[l,i,i] = HS_field[i,l]

        M_up = M_up_left @ expm(-K) @ expm(-nu*V_ls[l]) @ M_up_right
        M_dn = M_dn_left @ expm(-K) @ expm(-(-nu*V_ls[l])) @ M_dn_right

        det_M_up = np.linalg.det(I + M_up)
        det_M_dn = np.linalg.det(I + M_dn)
        M_after = det_M_up*det_M_dn
        
        P = M_after/M_before
    
        r = np.random.uniform(0,1)
        if r < P:
            pass
        else:
            HS_field[i,l] = HS_before
            V_ls[l,i,i] = HS_field[i,l]
    
    if sweep >= 100:
        I = np.eye(N)
        M_up = np.eye(N)
        M_dn = np.eye(N)
        for l_ in range(nt-1,-1,-1):
            V = np.zeros((N,N))
            np.fill_diagonal(V, HS_field[:,l_])
            
            M_up = np.dot(M_up, expm(-K))
            M_up = np.dot(M_up, expm(-nu*V))
            M_dn = np.dot(M_dn, expm(-K))
            M_dn = np.dot(M_dn, expm(-(-nu*V)))
    
        G_up = np.linalg.inv(I + M_up)
        G_dn = np.linalg.inv(I + M_dn)
        
        G_up = I - G_up
        G_dn = I - G_dn
        G_up_ls.append(G_up)
        G_dn_ls.append(G_dn)
        D_ls.append(np.mean(np.diag(G_up)*np.diag(G_dn)))

G_up_mean = np.mean(G_up_ls,axis=0)
G_dn_mean = np.mean(G_dn_ls,axis=0)
D_mean = np.mean(D_ls)
print()
print(D_mean)

plt.imshow(G_dn_mean)

