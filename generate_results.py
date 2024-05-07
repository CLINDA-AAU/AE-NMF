import os
import pandas as pd
import numpy as np
import torch
import timeit

from CNMF import *
from AENMF import * 
from NMF import *
from functions import cosine_HA, refit
from sklearn.model_selection import train_test_split


### Est C-NMF
# Takes a numpy array as input and estimates Convex NMF.
# Returns signatures, exposures, final loss, initial W and H matrices, and the number of iterations.
def est_cvx(X, rank, tol):
    G_cvx, W_cvx, loss_cvx, _, _, niter = convex_nmf_tol(X, rank, tol = tol, relative_tol = True, mse = True) # NB !!!!!!!!!!!!!!!!!!!!
    signatures_cvx = X@(W_cvx)
    diagonals_cvx = signatures_cvx.sum(axis = 0)
    exposures_cvx =  (G_cvx@np.diag(diagonals_cvx)).T
    signatures_cvx = (signatures_cvx)@np.diag(1/diagonals_cvx)
    return(signatures_cvx, exposures_cvx, loss_cvx[-1], niter)

### Est AE-NMF
# Takes a pandas DataFrame as input and estimates NMF-AE.
# Returns signatures, exposures, final loss, initial encoder and decoder matrices, and the number of iterations.
def est_AE(X, rank, tol):
    _, n = X.shape
    model_AE = AE_NMF_abs(patient_dim = n, latent_dim = rank)

    optimizer = torch.optim.Adam(params = model_AE.parameters(), lr=1e-4)

    criterion = torch.nn.MSELoss(reduction = 'mean') 
   
    _, loss_AE,signatures_AE, exposures_AE, _, niter = train_AENMF_tol(model = model_AE, x_train = X, criterion = criterion, optimizer = optimizer, tol = tol, relative_tol = True, constr = "fpabs")
    
    if np.any((signatures_AE==0).all(axis=0)):
        raise ValueError("You have a basis vector of all zeros")

    
    diagonals_AE = signatures_AE.sum(axis = 0)
    exposures_AE = np.diag(diagonals_AE)@exposures_AE
    signatures_AE = (signatures_AE)@np.diag(1/diagonals_AE)
    return(signatures_AE, exposures_AE, loss_AE[-1], niter)
    

### Est NMF (Idas version)
# Takes a numpy array as input and estimates NMF.
# Returns signatures, exposures, final loss, and the number of iterations.
def est_nmf(X, rank, tol):
    signatures_NMF, exposures_NMF, loss_NMF, _, _, niter = NMF_mult_tol(X = X, rank = rank, tol = tol, mse = True) 
    loss = loss_NMF[-1]
    diagonals_NMF = signatures_NMF.sum(axis = 0)
    exposures_NMF = np.diag(diagonals_NMF)@exposures_NMF
    signatures_NMF = signatures_NMF@np.diag(1/diagonals_NMF)

    return(signatures_NMF, exposures_NMF, loss, niter)
    
# Function to run all methods for each diagnosis
def run_all(data_df, diagnosis_list, rank_list, i, tol):
    res = []
    for j,diagnosis in enumerate(diagnosis_list):
      
      #save results
      dir_name = "generated_data/" +  diagnosis
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
      data = data_df[diagnosis]
      rank = rank_list[j]
      train,test = train_test_split(data.T, train_size = 0.8)
      train = train.T
      test = test.T

      # Estimating Convex NMF
      start = timeit.default_timer()
      signatures_cvx, exposures_cvx, loss_cvx, niter_cvx = est_cvx(train.to_numpy(), rank = rank, tol = tol)
      out_error_cvx = refit(test, signatures_cvx)
      stop = timeit.default_timer()
      time_cvx = stop - start
      print("CVX finished for diagnosis " + diagnosis + " it took " + str(time_cvx) + " seconds")
  
  
      # Estimating AE-NMF
      start = timeit.default_timer()
      signatures_AE, exposures_AE, loss_AE, niter_AE = est_AE(train, rank = rank, tol = tol)
      out_error_AE = refit(test, signatures_AE)
      stop = timeit.default_timer()
      time_AE = stop - start
      print("AE finished for diagnosis " + diagnosis + " it took " + str(time_AE) + " seconds")

    
      # Estimating NMF
      start = timeit.default_timer()
      signatures_nmf, exposures_nmf, loss_nmf, niter_nmf = est_nmf(train.to_numpy(), rank = rank, tol = tol)
      out_error_nmf = refit(test, signatures_nmf)
      stop = timeit.default_timer()
      time_NMF = stop - start
      print("NMF finished for diagnosis " + diagnosis +  " it took " + str(time_NMF) + " seconds")
      
      # Save estimated matrices
      pd.DataFrame(signatures_cvx).to_csv(dir_name + "/cvx_" + str(i) + "_sigs.csv")
      pd.DataFrame(exposures_cvx).to_csv(dir_name + "/cvx_" + str(i) + "_exp.csv")
      
      pd.DataFrame(signatures_AE).to_csv(dir_name + "/AEFPabs_" + str(i) + "_sigs.csv") 
      pd.DataFrame(exposures_AE).to_csv(dir_name + "/AEFPabs_" + str(i) + "_exp.csv")

      pd.DataFrame(signatures_nmf).to_csv(dir_name + "/nmf_" + str(i) + "_sigs.csv")
      pd.DataFrame(exposures_nmf).to_csv(dir_name + "/nmf_" + str(i) + "_exp.csv")



      # Cosine similarity calculations
      cosine_cvx_ae, _ = cosine_HA(signatures_cvx.T, signatures_AE.T)
      cosine_cvx_nmf, _ = cosine_HA(signatures_cvx.T, signatures_nmf.T)
      cosine_ae_nmf, _ = cosine_HA(signatures_AE.T, signatures_nmf.T)
      
      cosine_cvx_ae = np.mean(cosine_cvx_ae.diagonal())
      cosine_cvx_nmf = np.mean(cosine_cvx_nmf.diagonal())
      cosine_ae_nmf = np.mean(cosine_ae_nmf.diagonal())
      
      res.append([diagnosis, 
                  cosine_cvx_ae, cosine_cvx_nmf, cosine_ae_nmf, 
                  loss_cvx, loss_AE, loss_nmf,
                  out_error_cvx, out_error_AE, out_error_nmf,
                  time_cvx, time_AE, time_NMF,
                  niter_cvx, niter_AE, niter_nmf])
    return(res)

# Generate random data matrices
count_ovary = pd.DataFrame(np.random.rand(96,532))#pd.read_csv('~/WGS_data/catalogues_Ovary_SBS.tsv', sep = '\t')
count_prostate = pd.DataFrame(np.random.rand(96,311))#pd.read_csv('~/WGS_data/catalogues_Prostate_SBS.tsv', sep = '\t')
count_uterus = pd.DataFrame(np.random.rand(96,713))#pd.read_csv('~/WGS_data/catalogues_Uterus_SBS.tsv', sep = '\t')

diagnosis = ["Ovary", "Prostate", "Uterus"]
df_list = [ count_ovary, count_prostate, count_uterus]
df_dict = {diagnosis[i]: df_list[i] for i in range(len(diagnosis))}

n_sims = 30
rank_list = [4, 5, 8]
tol = 1e-10


#unfolding the data
res =[run_all(df_dict, diagnosis, rank_list, i = i, tol = tol) for i in range(n_sims)]

res1 = [np.vstack((r[0], r[1], r[2])) for r in res]

res2 = np.vstack(([r[0] for r in res1], 
                  [r[1] for r in res1],  
                  [r[2] for r in res1]))

result = pd.DataFrame(res2)



