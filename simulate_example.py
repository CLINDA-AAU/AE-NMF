import pandas as pd
import numpy as np
import torch
import timeit

# Importing functions from custom modules
from CNMF import convex_nmf_tol
from NMF import NMF_mult_tol
from AENMF import *
from functions import cosine_HA

import matplotlib
import matplotlib.pyplot as plt

n = 15
tol = 1e-10

# ----------------------- Data generation ------------------------

# Generating signature matrix and exposures
signature1 = np.array([2]*2 + [1]*2 + [0]*2)/6
signature2 = np.array([0]*3 + [2]*3)/6
signature_mat = np.vstack((signature1, signature2))
total_n = 3*n
exposures = np.array([190, 10]*n + [100,100]*n+ [20, 180]*n)

XMat1 = np.random.poisson(190*signature1 + 10*signature2, size = (n,6))
XMat2 = np.random.poisson(100*signature1 + 100*signature2, size = (n,6))
XMat3 = np.random.poisson(20*signature1 + 180*signature2, size = (n,6))
XMat = np.concatenate((XMat1, XMat2, XMat3))

# -------------------------- CVX NMF ------------------------------

XMat_pd = pd.DataFrame(XMat)
XMat_pd = XMat_pd.T

start = timeit.default_timer()

# Applying Convex NMF
G_cvx, W_cvx, loss_cvx, dec, enc, n_iter_cvx = convex_nmf_tol(XMat_pd.to_numpy(), 2, tol=tol, mse=True)

# Calculating signatures and exposures for Convex NMF
signatures_cvx = XMat_pd.to_numpy() @ (W_cvx)
diagonals_cvx = signatures_cvx.sum(axis=0)
exposures_cvx = G_cvx @ np.diag(diagonals_cvx)
signatures_cvx = (signatures_cvx) @ np.diag(1 / diagonals_cvx)

stop = timeit.default_timer()

print('Time cvx: ', stop - start)

# -------------------------- NMF ------------------------------

XMat_pd = pd.DataFrame(XMat)
XMat_pd = XMat_pd.T

start = timeit.default_timer()

# Applying NMF
signatures_nmf, exposures_nmf, loss_nmf, _, _, n_iter_nmf = NMF_mult_tol(XMat_pd.to_numpy(), 2, tol=tol, mse=True, objective_type = "frobenius", G_0 = dec.T)

# Calculating signatures and exposures for NMF
diagonals_nmf = signatures_nmf.sum(axis=0)
exposures_nmf = exposures_nmf.T @ np.diag(diagonals_nmf)
signatures_nmf = (signatures_nmf) @ np.diag(1 / diagonals_nmf)

stop = timeit.default_timer()

print('Time nmf: ', stop - start)

# ------------------------- AE NMF -----------------------------------
# Creating Autoencoder model
model_AE = AE_NMF_abs(patient_dim=total_n, latent_dim=2, start_enc = enc, start_dec = dec.T)
optimizer = torch.optim.Adam(params=model_AE.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss(reduction='mean')

start = timeit.default_timer()

# Training Autoencoder model
_, loss_AE, signatures_AE, exposures_AE, _, n_iter_AE = train_AENMF_tol(model=model_AE, x_train=XMat_pd,
                                                                      criterion=criterion, optimizer=optimizer,
                                                                      tol=tol, constr="fpabs")
# Calculating signatures and exposures for Autoencoder
diagonals_AE = signatures_AE.sum(axis=0)
exposures_AE = np.diag(diagonals_AE) @ exposures_AE
signatures_AE = (signatures_AE) @ np.diag(1 / diagonals_AE)

stop = timeit.default_timer()

print('Time AE: ', stop - start)
print((loss_AE[-1], loss_cvx[-1], loss_nmf[-1]))

# Reorder signatures and exposures to match
cosine_cvx, idx_CVX = cosine_HA(signature_mat, signatures_cvx.T)
signatures_cvx = signatures_cvx[:, idx_CVX]
exposures_cvx = exposures_cvx[:, idx_CVX]

cosine_nmf, idx_NMF = cosine_HA(signature_mat, signatures_nmf.T)
signatures_nmf = signatures_nmf[:, idx_NMF]
exposures_nmf = exposures_nmf[:, idx_NMF]

cosine_ae, idx_AE = cosine_HA(signature_mat, signatures_AE.T)
signatures_AE = signatures_AE[:, idx_AE]
exposures_AE = exposures_AE[idx_AE, :]




matplotlib.rcParams['figure.figsize'] = [20, 15]
matplotlib.rcParams.update({'font.size': 22})

fig1, axs1 = plt.subplots(2,2, width_ratios = [1.7,1])
plt.subplots_adjust(right=0.8)

axs1[0,0].plot(list(range(1,total_n+1)), [190]*n + [100]*n + [20]*n, '-o', color = "#c8ae95", label = "True", linewidth=4)
axs1[0,0].plot(list(range(1,total_n+1)), exposures_nmf[:,0], '-o', color = "#a3a09e", label = "NMF", linewidth=4)
axs1[0,0].plot(list(range(1,total_n+1)), exposures_AE[0,:], '-o', color = "#434961", label = "AE-NMF", linewidth=4)
axs1[0,0].plot(list(range(1,total_n+1)), exposures_cvx[:,0], ':o', color = "#d1b2d2", label = "C-NMF", linewidth=4)

axs1[0,0].set_xticklabels([])
axs1[0,0].xaxis.set_ticks_position('none') 

axs1[1,0].plot(list(range(1,total_n+1)), [10]*n + [100]*n + [180]*n, '-o', color = "#c8ae95", label = "True", linewidth=4)
axs1[1,0].plot(list(range(1,total_n+1)), exposures_nmf[:,1], '-o', color = "#a3a09e", label = "NMF", linewidth=4)
axs1[1,0].plot(list(range(1,total_n+1)), exposures_AE[1,:], '-o', color = "#434961", label = "AE-NMF", linewidth=4)
axs1[1,0].plot(list(range(1,total_n+1)), exposures_cvx[:,1], ':o', color = "#d1b2d2", label = "C-NMF", linewidth=4)
axs1[1,0].set_xticklabels([])
axs1[1,0].xaxis.set_ticks_position('none') 


axs1[0,1].plot(list(range(1, 7)), signature1, '-o', color = "#c8ae95", label = "True", linewidth=4)
axs1[0,1].plot(list(range(1, 7)), signatures_nmf[:,0], '-o', color = "#a3a09e", label = "NMF", linewidth=4)
axs1[0,1].plot(list(range(1, 7)), signatures_AE[:,0], '-o', color = "#434961", label = "AE-NMF", linewidth=4)
axs1[0,1].plot(list(range(1, 7)), signatures_cvx[:,0], ':o', color = "#d1b2d2", label = "C-NMF", linewidth=4)
axs1[0,1].set_xticklabels([])
axs1[0,1].xaxis.set_ticks_position('none') 


axs1[1,1].plot(list(range(1, 7)), signature2, '-o', color = "#c8ae95", label = "True", linewidth=4)
axs1[1,1].plot(list(range(1, 7)), signatures_nmf[:,1], '-o', color = "#a3a09e" , label = "NMF", linewidth=4)
axs1[1,1].plot(list(range(1, 7)), signatures_AE[:,1], '-o', color = "#434961" , label = "AE-NMF", linewidth=4)
axs1[1,1].plot(list(range(1, 7)), signatures_cvx[:,1], ':o', color = "#d1b2d2" , label = "C-NMF", linewidth=4)
axs1[1,1].set_xticklabels([])
axs1[1,1].xaxis.set_ticks_position('none') 


axs1[0,0].set_title("Weights of $h_{1}$")

axs1[1,0].set_title("Weights of $h_{2}$")

axs1[0,1].set_title("$h_{1}$")
axs1[1,1].set_title("$h_{2}$")


axs1[1,0].set_xlabel("Sample")
axs1[1,1].set_xlabel("Feature")

fig1.subplots_adjust(bottom=0.12)
fig1.legend(labels = ["True","NMF", "AE-NMF", "C-NMF"], loc="lower center", ncol=4)

plt.show()
