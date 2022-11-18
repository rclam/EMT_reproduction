#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:12:59 2022

@author: rclam
"""


# modules
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
#       Variable Set-Up
# ============================================================================
print("\n*** Setting up variables ***\n")
# scalar crack density (rho)
rho_a = 0.1
# theta (deg)
theta = np.array([0, 30, 45, 60, 90])
# normal tensors for each theta
a_n = np.zeros((5,3))
a_n[0] = -1, 0, 0
a_n[1] = -.5*(3**0.5), 0.5, 0
a_n[2] = -0.5*(2**0.5), 0.5*(2**0.5), 0
a_n[3] = -0.5, 0.5*(3**0.5), 0
a_n[4] = 0, 1, 0
print("n tensors: \n", a_n)

#   Kronecker Delta
Kronecker = np.zeros((3,3))
Kronecker[0,0] = 1.0; Kronecker[1,1] = 1.0; Kronecker[2,2] = 1.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Alpha Set-Up
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# theta = 0 deg.
a_alpha_0 = rho_a * np.outer(a_n[0], a_n[0])
print("\n\u03B1 (\u03B8 = 0\u00B0): \n", a_alpha_0)
# theta = 30 deg.
a_alpha_1 = rho_a * np.outer(a_n[1], a_n[1])
print("\n\u03B1 (\u03B8 = 30\u00B0): \n", a_alpha_1)
# theta = 45 deg.
a_alpha_2 = rho_a * np.outer(a_n[2], a_n[2])
print("\n\u03B1 (\u03B8 = 45\u00B0): \n", a_alpha_2)
# theta = 60 deg.
a_alpha_3 = rho_a * np.outer(a_n[3], a_n[3])
print("\n\u03B1 (\u03B8 = 60\u00B0): \n", a_alpha_3)
# theta = 90 deg.
a_alpha_4 = rho_a * np.outer(a_n[4], a_n[4])
print("\n\u03B1 (\u03B8 = 90\u00B0): \n", a_alpha_4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Intact Properties (Stiffness c_i and Compliance S_i)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# intact stiffness c_i (from healy)
c_i = np.zeros((6,6))
# row 1
c_i[0,0] = 38.83
c_i[0,1] = 4.80
c_i[0,2] = 4.80
# row 2
c_i[1,0] = 4.80
c_i[1,1] = 38.83
c_i[1,2] = 4.80
# row 3
c_i[2,0] = 4.80
c_i[2,1] = 4.80
c_i[2,2] = 38.83
# diagonal for rows 4-6
c_i[3,3] = 17.02
c_i[4,4] = 17.02
c_i[5,5] = 17.02
print("\nIntact stiffness c_i [GPa]: \n", c_i)

S_i = np.linalg.inv(c_i) 
print("\nIntact compliance Tensor S_i (by inversion): \n", S_i)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Stress and Pore Fluid Pressure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original Stress [MPa]
sigma = np.zeros((3,3))
sigma[0,0] = 87# 73 #87 #[MPa] sigma_1
sigma[2,2] = 40 #[MPa] sigma_3
sigma[1,1] = 0.5*(sigma[0,0] + sigma[2,2])
sigma_diff = 0.5*(abs(sigma[0,0] - sigma[2,2]))

# Pore Fluid Pressure
P_f = 50 # [MPa]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Solve for Elastic Moduli
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print("\n*** Solving for Elastic Moduli (parallel) ***\n")

# Solve for elastic modulii
lambda_i = c_i[0,1]
mu_i     = 0.5* (c_i[0,0] - lambda_i)
# print('c_12 = \u03BB = ', lambda_i)
# print('\u03BC        = ', mu_i)

E0 = mu_i * ((3*lambda_i  +  2*mu_i)/(lambda_i + mu_i))
# print('\nE0 = ', E0)
v = lambda_i/(2*(lambda_i + mu_i))
# print("\u03BD = ", v)
G0 = E0/(2*(1+v))
# print("\nElastic modulii [GPa]: \nYoung's Modulus E =", E0, " GPa\nPoisson ratio   \u03BD = ", v, "\nShear Modulus   G = ", G0," GPa\n")


# ============================================================================
#       Compliance Correction Term
# ============================================================================
print("\n*** Solve for compliance correction term for each ***\n")

delta_s_a = (8*(1-v**2))/(3*E0*(2-v))

#   Voigt Notation: Compliance Correction Term
delta_s_b_0 = np.zeros((6,6))
delta_s_b_1 = np.zeros((6,6))
delta_s_b_2 = np.zeros((6,6))
delta_s_b_3 = np.zeros((6,6))
delta_s_b_4 = np.zeros((6,6))

def Delta_S_b(delta_s_b_num, a_alpha_num):

    #   Row 1
    delta_s_b_num[0,0] = 4 * a_alpha_num[0,0]
    # delta_s_b[0,1] = 0
    # delta_s_b[0,2] = 0
    # delta_s_b[0,3] = 0
    delta_s_b_num[0,4] = 2 * a_alpha_num[0,2]
    delta_s_b_num[0,5] = 2 * a_alpha_num[0,1]
    
    #   Row 2
    # delta_s_b[1,0] = 0
    delta_s_b_num[1,1] = 4 * a_alpha_num[1,1]
    # delta_s_b[1,2] = 0
    delta_s_b_num[1,3] = 2 * a_alpha_num[1,2]
    # delta_s_b[1,4] = 0
    delta_s_b_num[1,5] = 2 * a_alpha_num[1,0]
    
    #   Row 3
    # delta_s_b[2,0] = 0
    # delta_s_b[2,1] = 0
    delta_s_b_num[2,2] = 4 * a_alpha_num[2,2]
    delta_s_b_num[2,3] = 2 * a_alpha_num[2,1]
    delta_s_b_num[2,4] = 2 * a_alpha_num[2,0]
    # delta_s_b[2,5] = 0
    
    #   Row 4
    # delta_s_b[3,0] = 0
    delta_s_b_num[3,1] = 2 * a_alpha_num[2,1]
    delta_s_b_num[3,2] = 2 * a_alpha_num[1,2]
    delta_s_b_num[3,3] = a_alpha_num[1,1] + a_alpha_num[2,2]
    delta_s_b_num[3,4] = a_alpha_num[1,0]
    delta_s_b_num[3,5] = a_alpha_num[2,0]
    
    #   Row 5
    delta_s_b_num[4,0] = 2 * a_alpha_num[2,0]
    # delta_s_b[4,1] = 0
    delta_s_b_num[4,2] = 2 * a_alpha_num[0,2]
    delta_s_b_num[4,3] = a_alpha_num[0,1]
    delta_s_b_num[4,4] = a_alpha_num[0,0] + a_alpha_num[2,2]
    delta_s_b_num[4,5] = a_alpha_num[2,1]
    
    #   Row 6
    delta_s_b_num[5,0] = 2 * a_alpha_num[1,0]
    delta_s_b_num[5,1] = 2 * a_alpha_num[0,1]
    # delta_s_b[5,2] = 0
    delta_s_b_num[5,3] = a_alpha_num[0,2]
    delta_s_b_num[5,4] = a_alpha_num[1,2]
    delta_s_b_num[5,5] = a_alpha_num[0,0] + a_alpha_num[1,1]
    return delta_s_b_num


S_voight_0 = delta_s_a * Delta_S_b(delta_s_b_0, a_alpha_0)
print("\n \u0394S (S_correction term \u03B8 = 0\u00B0): \n", S_voight_0)

S_voight_1 = delta_s_a * Delta_S_b(delta_s_b_1, a_alpha_1)
print("\n \u0394S (S_correction term \u03B8 = 30\u00B0): \n", S_voight_1)

S_voight_2 = delta_s_a * Delta_S_b(delta_s_b_2, a_alpha_2)
print("\n \u0394S (S_correction term \u03B8 = 45\u00B0): \n", S_voight_2)

S_voight_3 = delta_s_a * Delta_S_b(delta_s_b_3, a_alpha_3)
print("\n \u0394S (S_correction term \u03B8 = 60\u00B0): \n", S_voight_3)

S_voight_4 = delta_s_a * Delta_S_b(delta_s_b_4, a_alpha_4)
print("\n \u0394S (S_correction term \u03B8 = 90\u00B0): \n", S_voight_4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Solve for new Compliance S_e
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEW Compliance S_e
S_e_0 = S_i + S_voight_0
print("S_e 0 [GPa]: \n", S_e_0)

S_e_1 = S_i + S_voight_1
print("S_e 1 [GPa]: \n", S_e_1)

S_e_2 = S_i + S_voight_2
print("S_e 2 [GPa]: \n", S_e_2)

S_e_3 = S_i + S_voight_3
print("S_e 3 [GPa]: \n", S_e_3)

S_e_4 = S_i + S_voight_4
print("S_e 4 [GPa]: \n", S_e_4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       Invert for new CRACKED Stiffness c
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c_0 = np.linalg.inv(S_e_0) 
print("\nStiffness Tensor c0 (cracked) [GPa]: \n", c_0)

c_1 = np.linalg.inv(S_e_1) 
print("\nStiffness Tensor c1 (cracked) [GPa]: \n", c_1)

c_2 = np.linalg.inv(S_e_2) 
print("\nStiffness Tensor c2 (cracked) [GPa]: \n", c_2)

c_3 = np.linalg.inv(S_e_3) 
print("\nStiffness Tensor c3 (cracked) [GPa]: \n", c_3)

c_4 = np.linalg.inv(S_e_4) 
print("\nStiffness Tensor c4 (cracked) [GPa]: \n", c_4)


# ============================================================================
#       Solve for Biot Tensor
# ============================================================================
print("\n\n*** Solving for Biot tensor \u03B2_{ij} ***\n\u03B2_{ij} = \u03B4_{ij} - c_{ijkl} * s_{klmm}\n")

S_iv_1 = S_i[0,0] + S_i[0,1] + S_i[0,2]
S_iv_2 = S_i[1,0] + S_i[1,1] + S_i[1,2]
S_iv_3 = S_i[2,0] + S_i[2,1] + S_i[2,2]

CS_0 = np.zeros((3,3))
CS_1 = np.zeros((3,3))
CS_2 = np.zeros((3,3))
CS_3 = np.zeros((3,3))
CS_4 = np.zeros((3,3))

def CS_calc(CS_num, c_num):
# row 1
    CS_num[0,0] = c_num[0,0]*(S_iv_1) + c_num[0,1]*(S_iv_2) + c_num[0,2]*(S_iv_3)   # CS 1
    CS_num[0,1] = c_num[5,0]*(S_iv_1) + c_num[5,1]*(S_iv_2) + c_num[5,2]*(S_iv_3)   # CS 6
    CS_num[0,2] = c_num[4,0]*(S_iv_1) + c_num[4,1]*(S_iv_2) + c_num[4,2]*(S_iv_3)   # CS 5
    # row 2
    CS_num[1,0] = CS_num[0,1]                                               #      CS 6
    CS_num[1,1] = c_num[1,0]*(S_iv_1) + c_num[1,1]*(S_iv_2) + c_num[1,2]*(S_iv_3)   # CS 2
    CS_num[1,2] = c_num[3,0]*(S_iv_1) + c_num[3,1]*(S_iv_2) + c_num[3,2]*(S_iv_3)   # CS 4
    # row 3
    CS_num[2,0] = CS_num[0,2]                                               #      CS 5
    CS_num[2,1] = CS_num[1,2]                                               #      CS 4
    CS_num[2,2] = c_num[2,0]*(S_iv_1) + c_num[2,1]*(S_iv_2) + c_num[2,2]*(S_iv_3)   # CS 3
    
    return CS_num
# print("CS [no units]: \n", CS)

CS_0 = CS_calc(CS_0, c_0)
# CS_0[0,0] = 1.02 #1.3   # solve by work backwd by hand
# CS_0[2,2] = 0.64


CS_1 = CS_calc(CS_1, c_1)
# CS_1[0,0] = 0.96 #1.24   # solve by work backwd by hand
# CS_1[2,2] = 0.7

CS_2 = CS_calc(CS_2, c_2)
# CS_2[0,0] = 0.86 #1.14   # solve by work backwd by hand
# CS_2[2,2] = 0.8

CS_3 = CS_calc(CS_3, c_3)
# CS_3[0,0] = 0.76 #1.04   # solve by work backwd by hand
# CS_3[2,2] = 0.9

CS_4 = CS_calc(CS_4, c_4)
# CS_4[0,0] = 0.66 #0.94   # solve by work backwd by hand
# CS_4[2,2] = 0.96

# Biot = Kronecker - CS
Biot_0 = Kronecker - CS_0
print("\nCS 0: \n", CS_0)


print("\nBiot Tensor 0 [no units]: \n", Biot_0)

Biot_1 = Kronecker - CS_1
print("\n\nCS 1: \n", CS_1)
print("\nBiot Tensor 1 [no units]: \n", Biot_1)

Biot_2 = Kronecker - CS_2
print("\n\nCS 2: \n", CS_2)
print("\nBiot Tensor 2 [no units]: \n", Biot_2)

Biot_3 = Kronecker - CS_3
print("\n\nCS 3: \n", CS_3)
print("\nBiot Tensor 3 [no units]: \n", Biot_3)

Biot_4 = Kronecker - CS_4
print("\n\nCS 4: \n", CS_4)
print("\nBiot Tensor 4 [no units]: \n", Biot_4)



# ============================================================================
#       Solve for Effective Stresses
# ============================================================================
print("\n*** Solving for effective stress ***\n         \u03C3' = \u03C3 - Pf*\u03B2_{ij}")


def Effective_Stress(sigma, Biot, P_f=50):
    sigma_eff = (sigma) - P_f*(Biot)
    # print("\nEffective stress [MPa]: \n", sigma_eff)
    return sigma_eff

def Effective_Diff_Stress(sigma_eff):
    sigma_eff_diff = abs(sigma_eff[0,0] - sigma_eff[2,2])
    # print("\nDifferential \u03C3 [MPa]: ", sigma_eff_diff)
    return sigma_eff_diff

sigma_eff_0 = Effective_Stress(sigma, Biot_0)
sigma_eff_1 = Effective_Stress(sigma, Biot_1)
sigma_eff_2 = Effective_Stress(sigma, Biot_2)
sigma_eff_3 = Effective_Stress(sigma, Biot_3)
sigma_eff_4 = Effective_Stress(sigma, Biot_4)

sigma_eff_diff_0 = Effective_Diff_Stress(sigma_eff_0)
sigma_eff_diff_1 = Effective_Diff_Stress(sigma_eff_1)
sigma_eff_diff_2 = Effective_Diff_Stress(sigma_eff_2)
sigma_eff_diff_3 = Effective_Diff_Stress(sigma_eff_3)
sigma_eff_diff_4 = Effective_Diff_Stress(sigma_eff_4)

# ============================================================================
#       Plot Graph
# ============================================================================

sigma_1 = np.array([sigma_eff_0[0,0], sigma_eff_1[0,0], sigma_eff_2[0,0], sigma_eff_3[0,0], sigma_eff_4[0,0]])
sigma_3 = np.array([sigma_eff_0[2,2], sigma_eff_1[2,2], sigma_eff_2[2,2], sigma_eff_3[2,2], sigma_eff_4[2,2]])
sigma_diff = np.array([sigma_eff_diff_0, sigma_eff_diff_1, sigma_eff_diff_2, sigma_eff_diff_3, sigma_eff_diff_4])

plt.figure(1)
"""
# theta 0
plt.plot(theta[0], sigma_eff_0[0,0], 'k*', label='$\sigma$_1')
plt.plot(theta[0], sigma_eff_0[2,2], 'r*', label='$\sigma$ 3')
plt.plot(theta[0], (sigma_eff_diff_0), 'g*', label='$\sigma$ diff')
# theta 30
plt.plot(theta[1], sigma_eff_1[0,0], 'k*')
plt.plot(theta[1], sigma_eff_1[2,2], 'r*')
plt.plot(theta[1], (sigma_eff_diff_1), 'g*')
# theta 45
plt.plot(theta[2], sigma_eff_2[0,0], 'k*')
plt.plot(theta[2], sigma_eff_2[2,2], 'r*')
plt.plot(theta[2], (sigma_eff_diff_2), 'g*')
# theta 60
plt.plot(theta[3], sigma_eff_3[0,0], 'k*')
plt.plot(theta[3], sigma_eff_3[2,2], 'r*')
plt.plot(theta[3], (sigma_eff_diff_3), 'g*')
# theta 90
plt.plot(theta[4], sigma_eff_4[0,0], 'k*')
plt.plot(theta[4], sigma_eff_4[2,2], 'r*')
plt.plot(theta[4], (sigma_eff_diff_4), 'g*')
"""
plt.plot(90-theta, sigma_1, 'k-', label='$\sigma$_1')
plt.plot(theta, sigma_3, 'r-', label='$\sigma$_3')
plt.plot(theta, sigma_diff, 'g-', label='$\sigma$ diff')



plt.xlabel('Theta, degrees')
plt.ylabel('Effective stress, MPa')
# plt.legend(loc='center left')
plt.legend(loc='lower right')
plt.title('Crack density = 0.1')
plt.grid()
plt.ylim([0,100])
plt.xlim([-0.01,100])
plt.show()



