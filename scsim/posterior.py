import copy
import os
import logging

from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import torch.distributions as distributions

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    SubsetRandomSampler,
    RandomSampler,
)

from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scsim import GeneExpressionDataset
logger = logging.getLogger(__name__)



class Posterior:
    r"""The functional data unit. A `Posterior` instance is instantiated with a model and a gene_dataset, and
    as well as additional arguments that for Pytorch's `DataLoader`. A subset of indices can be specified, for
    purposes such as splitting the data into train/test or labelled/unlabelled (for semi-supervised learning).
    Each trainer instance of the `Trainer` class can therefore have multiple `Posterior` instances to train a model.
    A `Posterior` instance also comes with many methods or utilities for its corresponding data.


    :param model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
    :param gene_dataset: A gene_dataset instance like ``CortexDataset()``
    :param shuffle: Specifies if a `RandomSampler` or a `SequentialSampler` should be used
    :param indices: Specifies how the data should be split with regards to train/test or labelled/unlabelled
    :param use_cuda: Default: ``True``
    :param data_loader_kwarg: Keyword arguments to passed into the `DataLoader`

    Examples:

    Let us instantiate a `trainer`, with a gene_dataset and a model

        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels, use_cuda=True)
        >>> trainer = UnsupervisedTrainer(vae, gene_dataset)
        >>> trainer.train(n_epochs=50)

    A `UnsupervisedTrainer` instance has two `Posterior` attributes: `train_set` and `test_set`
    For this subset of the original gene_dataset instance, we can examine the differential expression,
    log_likelihood, entropy batch mixing, ... or display the TSNE of the data in the latent space through the
    scVI model

        >>> trainer.train_set.differential_expression_stats()
        >>> trainer.train_set.reconstruction_error()
        >>> trainer.train_set.entropy_batch_mixing()
        >>> trainer.train_set.show_t_sne(n_samples=1000, color_by="labels")

    """

    def __init__(
        self,
        model,
        gene_dataset: GeneExpressionDataset,
        shuffle=False,
        indices=None,
        use_cuda=True,
        data_loader_kwargs=dict(),
    ):
        """

        When added to annotation, has a private name attribute
        """
        self.model = model
        self.gene_dataset = gene_dataset
        self.use_cuda = use_cuda

        if indices is not None and shuffle:
            raise ValueError("indices is mutually exclusive with shuffle")
        if indices is None:
            if shuffle:
                sampler = RandomSampler(gene_dataset)
            else:
                sampler = SequentialSampler(gene_dataset)
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            sampler = SubsetRandomSampler(indices)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        self.data_loader_kwargs.update(
            {"collate_fn": gene_dataset.collate_fn_builder(), "sampler": sampler}
        )
        self.data_loader = DataLoader(gene_dataset, **self.data_loader_kwargs)

    def accuracy(self):
        pass

    accuracy.mode = "max"

    def __iter__(self):
        return map(self.to_cuda, iter(self.data_loader))

    def to_cuda(self, tensors):
        return [t.cuda() if self.use_cuda else t for t in tensors]


    @torch.no_grad()
    def get_latent(self, sample=False):
        """
        Output posterior z mean or sample, batch index, and label
        :param sample: z mean or z sample
        :return: three np.ndarrays, latent, batch_indices, labels
        """
        latent = []
        batch_indices = []
        labels = []
        for tensors in self:
            
            sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
            give_mean = not sample
            latent += [
                self.model.sample_from_posterior_z(
                    sample_batch, give_mean=give_mean
                ).cpu()
            ]
            batch_indices += [batch_index.cpu()]
            labels += [label.cpu()]
        return (
            np.array(torch.cat(latent)),
            np.array(torch.cat(batch_indices)),
            np.array(torch.cat(labels)).ravel(),
        )
    def show_t_sne(
        self,
        n_samples=1000,
        color_by="",
        save_name="",
        latent=None,
        batch_indices=None,
        labels=None,
        n_batch=None,
        train_resample_measure_real=False,
    ):
        # If no latent representation is given
        if latent is None:
            latent, batch_indices, labels = self.get_latent(sample=True)
#            means=np.zeros((7,latent.shape[1]))
#            weights_init=[]
#            for i in range(7):
#                latent_i=latent[labels==i,:]
#                mean_i=np.mean(latent_i, axis=0)
#                means[i,:]=mean_i.reshape([latent.shape[1]])
#                weights_init.append(np.sum(labels==i))
#            weights_init=np.array(weights_init)
#            weights_init=weights_init/np.sum(weights_init)
            gmm1 = GaussianMixture(n_components=7,  covariance_type='tied', n_init = 10, max_iter = 300).fit(latent)
            labels_gmm1 = gmm1.predict(latent)
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
            labels_gmm1 = labels_gmm1[idx_t_sne]
            print(metrics.adjusted_rand_score(labels, labels_gmm1))
        else:
            gmm1 = GaussianMixture(n_components=7, covariance_type='tied', n_init = 10, max_iter = 300).fit(latent)
            if train_resample_measure_real:
                latent, batch_indices, labels = self.get_latent(sample=True)
            labels_gmm1 = gmm1.predict(latent)
            latent, idx_t_sne = self.apply_t_sne(latent, n_samples)
            batch_indices = batch_indices[idx_t_sne].ravel()
            labels = labels[idx_t_sne].ravel()
            labels_gmm1 = labels_gmm1[idx_t_sne]
            print(metrics.adjusted_rand_score(labels, labels_gmm1))


        if n_batch is None:
            n_batch = self.gene_dataset.n_batches

        if color_by == "batches" or color_by == "labels":
            indices = (
                batch_indices.ravel() if color_by == "batches" else labels.ravel()
            )
            n = n_batch if color_by == "batches" else self.gene_dataset.n_labels
            if self.gene_dataset.cell_types is not None and color_by == "labels":
                plt_labels = self.gene_dataset.cell_types
            else:
                plt_labels = [str(i) for i in range(len(np.unique(indices)))]
            plt.figure(figsize=(8, 8))
            for i, label in zip(range(n), plt_labels):
                plt.scatter(
                    latent[indices == i, 0], latent[indices == i, 1], label=label
                )
            plt.legend()


        plt.axis("off")
        plt.tight_layout()
        if save_name:
            plt.show()
            plt.savefig(save_name)



        if n_batch is None:
            n_batch = self.gene_dataset.n_batches

        if color_by == "batches" or color_by == "labels":
            indices = (
                batch_indices.ravel() if color_by == "batches" else labels_gmm1.ravel()
            )
            n = n_batch if color_by == "batches" else self.gene_dataset.n_labels
            if self.gene_dataset.cell_types is not None and color_by == "labels":
                plt_labels = self.gene_dataset.cell_types
            else:
                plt_labels = [str(i) for i in range(len(np.unique(indices)))]
            plt.figure(figsize=(8, 8))
            for i, label in zip(range(n), plt_labels):
                plt.scatter(
                    latent[indices == i, 0], latent[indices == i, 1], label=label
                )
            plt.legend()


        plt.axis("off")
        plt.tight_layout()
        if save_name:
            print('here')
            plt.show()
            plt.savefig(save_name)

    @staticmethod
    def apply_t_sne(latent, n_samples=1000):
        idx_t_sne = (
            np.random.permutation(len(latent))[:n_samples]
            if n_samples
            else np.arange(len(latent))
        )
        if latent.shape[1] != 2:
            latent = TSNE().fit_transform(latent[idx_t_sne])
        return latent, idx_t_sne

