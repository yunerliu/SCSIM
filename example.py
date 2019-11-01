import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scsim import  CortexDataset, Sampler
import torch
from sklearn.mixture import GaussianMixture as GM

#load dataset
save_path='data/'
dataset = CortexDataset(save_path=save_path, total_genes=2000) 

#train model and then sample new data
sampler= Sampler(dataset, n_label=7)
sampler.train()
n_sample=100
xs, labels=sampler.sample(n_sample)
