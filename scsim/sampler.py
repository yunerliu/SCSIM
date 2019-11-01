
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .cortex import  CortexDataset
from .vae import VAE
from .trainer import Trainer
import torch
from sklearn.mixture import GaussianMixture as GM

class Sampler: 
    
    def __init__(self, dataset, n_label, n_hidden_vae=128, n_latent_vae=10, use_batches=False):
        self.dataset=dataset
        self.n_label=n_label
        self.n_hidden_vae=n_hidden_vae
        self.n_latent_vae=n_latent_vae
        self.vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches, n_hidden = n_hidden_vae, n_latent = n_latent_vae)
        self.trainer=Trainer(self.vae, dataset, train_size=0.9)
        self.gm=None
        
    def train(self):
        n_epochs=250   
        self.trainer.train(n_epochs=n_epochs) 
        latent, batch_indices, labels = self.trainer.train_set.get_latent()        
        gm=GM(n_components=self.n_label, covariance_type='tied')
        gm.fit(latent)
        self.gm=gm
        return

    def sample(self, n_sample):
        use_cuda=torch.cuda.is_available()
        res = self.gm.sample(n_sample)
        
        latent_new, labels_new = res[0], res[1]       
        latent_new=torch.tensor(latent_new)
        if use_cuda==True:
            latent_new=latent_new.cuda()
            latent_new=latent_new.type(torch.cuda.FloatTensor)
        highD_samples=self.trainer.model.sample_from_posterior_x(latent_new)
        highD_samples_np=np.array(highD_samples.cpu())
        return (highD_samples_np, labels_new)

