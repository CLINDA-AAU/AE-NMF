# Importing necessary libraries

import numpy as np
import pandas as pd
import os
import glob
import itertools
from functions import *

# Specify the diagnosis for analysis
diagnosis = "Prostate"

# Directory for storing signature matrices
dir = "generated_data/" + diagnosis
os.chdir(dir)

# List all the signature files for NMF, AEFPabs, and Convex NMF
NMF_signature_files = list(glob.glob('nmf*sigs.csv'))
AEFPabs_signature_files = list(glob.glob('AEFPabs*sigs.csv'))
CVX_signature_files = list(glob.glob('cvx*sigs.csv'))

# Read signature files and store them in dictionaries
sigs_NMF = {(x.split('nmf_')[1]).split('_sigs')[0] + "_1" : (pd.read_csv(x, sep = ",", index_col = 0)).to_numpy() for x in NMF_signature_files}
sigs_AEfpabs = {(x.split('AEFPabs_')[1]).split('_sigs')[0] + "_1" : (pd.read_csv(x, sep = ",", index_col = 0)).to_numpy() for x in AEFPabs_signature_files}
sigs_CVX = {(x.split('cvx_')[1]).split('_sigs')[0] + "_1" : (pd.read_csv(x, sep = ",", index_col = 0)).to_numpy() for x in CVX_signature_files}


# Generate combinations of pairs of signatures
x = list(sigs_NMF.keys())
cosine_cvx = []
cosine_AEfpabs = []
cosine_NMF = []

print(len(x))
print(len(list(itertools.combinations(x, 2))))

# Calculate cosine similarity for each pair of signatures

for i1,i2 in itertools.combinations(x,2):
  sigs1_cvx = sigs_CVX[str(i1)]
  sigs2_cvx = sigs_CVX[str(i2)]
  cosine_cvx_mat, idx_cvx = cosine_HA(sigs1_cvx.T, sigs2_cvx.T)
  sigs2_cvx = sigs2_cvx[:, idx_cvx]
  cosine_cvx.append(np.mean(cosine_cvx_mat.diagonal()))
  
  sigs1_AEfpabs = sigs_AEfpabs[str(i1)]
  sigs2_AEfpabs = sigs_AEfpabs[str(i2)]
  cosine_AEfpabs_mat, idx_AEfpabs = cosine_HA(sigs1_AEfpabs.T, sigs2_AEfpabs.T)
  sigs2_AEfpabs = sigs2_AEfpabs[:, idx_AEfpabs]
  cosine_AEfpabs.append(np.mean(cosine_AEfpabs_mat.diagonal()))
  
  sigs1_NMF = sigs_NMF[str(i1)]
  sigs2_NMF = sigs_NMF[str(i2)]
  cosine_NMF_mat, idx_NMF = cosine_HA(sigs1_NMF.T, sigs2_NMF.T)
  sigs2_NMF = sigs2_NMF[:, idx_NMF]
  cosine_NMF.append(np.mean(cosine_NMF_mat.diagonal()))

# Create a DataFrame to store the cosine similarity for each method
cosine_cons_mat = pd.DataFrame(zip(cosine_cvx, cosine_AEfpabs, cosine_NMF), columns = ["CVX", "AEfpabs", "NMF"])
print(cosine_cons_mat)
  


