# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

class AE_NMF_pg(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization using Projected Gradient.

    Parameters:
    patient_dim (int): Dimension of input data.
    latent_dim (int): Dimension of latent space.

    Attributes:
    enc_weight (torch.nn.Parameter): Encoder weight matrix.
    dec_weight (torch.nn.Parameter): Decoder weight matrix.
    """
    def __init__(self, patient_dim, latent_dim, start_enc = None, start_dec = None):
        super().__init__()
        if start_enc is not None:
           self.enc_weight = torch.nn.Parameter(torch.Tensor(start_enc))
        else: 
           self.enc_weight = torch.nn.Parameter(torch.rand(patient_dim, latent_dim)) 
        
        if start_dec is not None:
           self.dec_weight = torch.nn.Parameter(torch.Tensor(start_dec))
        else: 
           self.dec_weight = torch.nn.Parameter(torch.rand(latent_dim, patient_dim))

    def forward(self, x):
        x = torch.matmul(x,  F.relu(self.enc_weight))
        x = torch.matmul(x, F.relu(self.dec_weight))
        return x


class AE_NMF_abs(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization using Absolute Values.

    Parameters:
    patient_dim (int): Dimension of input data.
    latent_dim (int): Dimension of latent space.

    Attributes:
    enc_weight (torch.nn.Parameter): Encoder weight matrix.
    dec_weight (torch.nn.Parameter): Decoder weight matrix.
    """

    def __init__(self, patient_dim, latent_dim, start_enc = None, start_dec = None):
    
        super().__init__()
        if start_enc is not None:
           self.enc_weight = torch.nn.Parameter(torch.Tensor(start_enc))
        else: 
           self.enc_weight = torch.nn.Parameter(torch.rand(patient_dim, latent_dim)) 
        
        if start_dec is not None:
           self.dec_weight = torch.nn.Parameter(torch.Tensor(start_dec))
        else: 
           self.dec_weight = torch.nn.Parameter(torch.rand(latent_dim, patient_dim))

    def forward(self, x):
        x = torch.matmul(x, torch.abs(self.enc_weight))
        x = torch.matmul(x, torch.abs(self.dec_weight))
        return x


class AE_NMF(torch.nn.Module):
    """
    Autoencoder for Non-negative Matrix Factorization.

    Parameters:
    patient_dim (int): Dimension of input data.
    latent_dim (int): Dimension of latent space.
    start_enc (numpy.ndarray): If given initial encoding matrix of size (patient_dim, latent_dim)(default is None)
    start_dec (numpy.ndarray): If given initial decoding matrix of size (latent_dim, patient_dim) (default is None)

        
    Attributes:
    enc_weight (torch.nn.Parameter): Encoder weight matrix.
    dec_weight (torch.nn.Parameter): Decoder weight matrix.
    """
    def __init__(self, patient_dim, latent_dim, start_enc = None, start_dec = None):
    
        super().__init__()
        if start_enc is not None:
           self.enc_weight = torch.nn.Parameter(torch.Tensor(start_enc))
        else: 
           self.enc_weight = torch.nn.Parameter(torch.rand(patient_dim, latent_dim)) 
        
        if start_dec is not None:
           self.dec_weight = torch.nn.Parameter(torch.Tensor(start_dec))
        else: 
           self.dec_weight = torch.nn.Parameter(torch.rand(latent_dim, patient_dim))
         
    def forward(self, x):
        x = torch.matmul(x,self.enc_weight)
        x = torch.matmul(x, self.dec_weight)
        return x
    

def train_AENMF_tol(model, x_train, criterion, optimizer, tol = 1e-3, relative_tol = True, max_iter = 10e8, constr = "pg"):
    """
    Trains the Autoencoder for Non-negative Matrix Factorization until a given tolerance level is reached.

    Parameters:
    model (torch.nn.Module): Autoencoder model.
    x_train (numpy.ndarray): Training data.
    criterion (torch.nn.Module): Loss criterion.
    optimizer (torch.optim.Optimizer): Optimizer.
    tol (float): Tolerance level for convergence (default is 1e-3).
    relative_tol (bool): If True, the tolerance is relative, otherwise absolute (default is True).
    max_iter (int): Maximum number of iterations (default is 10e8).
    constr (str): Constraint type, either "pg", "fppg", "abs or "fpabs" (default is "pg").

    Returns:
    model (torch.nn.Module): Trained autoencoder model.
    training_loss (list): List of training loss values at each iteration.
    signatures (numpy.ndarray): Signatures matrix.
    exposures_train (numpy.ndarray): Exposure matrix.
    enc_mat (numpy.ndarray): Encoder matrix.
    n_iter (int): Number of iterations performed.
    """
    x_train_tensor = torch.tensor(x_train.values, 
                              dtype = torch.float32)


    training_loss = [1e10]#[float('inf')]
    rel_diff = float('inf')
    n_iter = 0
    while rel_diff>tol:
      n_iter += 1 
      optimizer.zero_grad()

      # Output of Autoencoder
      reconstructed = model(x_train_tensor)
      loss = criterion(reconstructed, x_train_tensor)
      
      loss.backward()
      optimizer.step()
      loss_val = loss.item()
      training_loss.append(loss_val)
      
      denominator = training_loss[-2] if relative_tol else 1
      rel_diff = abs(training_loss[-1]-training_loss[-2])/denominator

      with torch.no_grad():
        if constr == "pg": #PG after update
          for p in model.parameters():
              p.clamp_(min = 0)
        if constr == "abs": #abs after update
          for p in model.parameters():
              p = torch.abs(p)
      if n_iter>=max_iter:
        break
      
    del training_loss[0]
    #constrain final signatures and exposures 

    if constr == "abs" or constr == "fpabs":
      enc_mat = torch.abs(model.enc_weight).data
      signatures = (x_train@enc_mat).to_numpy()
      exposures_train = (torch.abs(model.dec_weight).data).numpy()
      
    if constr == "pg" or constr == "fppg":
      enc_mat = (model.enc_weight.data).numpy().clip(min = 0, max = None)
      signatures = (x_train@enc_mat).to_numpy()
      exposures_train = (model.dec_weight.data).numpy().clip(min = 0, max = None)
        
    return (model, training_loss, signatures, exposures_train, enc_mat, n_iter)