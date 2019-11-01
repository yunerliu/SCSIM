import logging
import sys
import time

from abc import abstractmethod
from collections import defaultdict, OrderedDict
from itertools import cycle

import numpy as np
import torch

from sklearn.model_selection._split import _validate_shuffle_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from scsim import Posterior

logger = logging.getLogger(__name__)


class Trainer:
    r"""The abstract Trainer class for training a PyTorch model and monitoring its statistics. It should be
    inherited at least with a .loss() function to be optimized in the training loop.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :use_cuda: Default: ``True``.
        :metrics_to_monitor: A list of the metrics to monitor. If not specified, will use the
            ``default_metrics_to_monitor`` as specified in each . Default: ``None``.
        :benchmark: if True, prevents statistics computation in the training. Default: ``False``.
        :frequency: The frequency at which to keep track of statistics. Default: ``None``.
        :early_stopping_metric: The statistics on which to perform early stopping. Default: ``None``.
        :save_best_state_metric:  The statistics on which we keep the network weights achieving the best store, and
            restore them at the end of training. Default: ``None``.
        :on: The data_loader name reference for the ``early_stopping_metric`` and ``save_best_state_metric``, that
            should be specified if any of them is. Default: ``None``.
        :show_progbar: If False, disables progress bar.
        :seed: Random seed for train/test/validate split
        
        
The VariationalInference class for the unsupervised training of an autoencoder.

    Args:
        :model: A model instance from class ``VAE``, ``VAEC``, ``SCANVI``
        :gene_dataset: A gene_dataset instance like ``CortexDataset()``
        :train_size: The train size, either a float between 0 and 1 or an integer for the number of training samples
         to use Default: ``0.8``.
        :test_size: The test size, either a float between 0 and 1 or an integer for the number of training samples
         to use Default: ``None``, which is equivalent to data not in the train set. If ``train_size`` and ``test_size``
         do not add to 1 or the length of the dataset then the remaining samples are added to a ``validation_set``.
        :n_epochs_kl_warmup: Number of epochs for linear warmup of KL(q(z|x)||p(z)) term. After `n_epochs_kl_warmup`,
            the training objective is the ELBO. This might be used to prevent inactivity of latent units, and/or to
            improve clustering of latent space, as a long warmup turns the model into something more of an autoencoder.
        :\*\*kwargs: Other keywords arguments from the general Trainer class.

    Examples:
        >>> gene_dataset = CortexDataset()
        >>> vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * False,
        ... n_labels=gene_dataset.n_labels)

        >>> infer = VariationalInference(gene_dataset, vae, train_size=0.5)
        >>> infer.train(n_epochs=20, lr=1e-3)        
        
    """
    def __init__(
        self,
        model,
        gene_dataset,
        train_size=0.8,
        weight_decay=1e-6,
        seed=0,
    ):
        # handle mutable defaults
        self.model = model
        self.gene_dataset = gene_dataset
        self._posteriors = OrderedDict()
        self.seed = seed
        self.data_loader_kwargs = {"batch_size": 128, "pin_memory": True}
        self.weight_decay = weight_decay
        self.epoch = -1  # epoch = self.epoch + 1 in compute metrics
        self.training_time = 0
       
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.show_progbar = True
        test_size=None
        self.n_epochs_kl_warmup = 400
        self.train_set, self.test_set, self.validation_set = self.train_test_validation(
                model, gene_dataset, train_size, test_size
            )

    @torch.no_grad()
    def compute_metrics(self):
        begin = time.time()
        self.compute_metrics_time += time.time() - begin

    def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None):
        begin = time.time()
        self.model.train()

        if params is None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = self.optimizer = torch.optim.Adam(
            params, lr=lr, eps=eps, weight_decay=self.weight_decay
        )

        self.compute_metrics_time = 0
        self.n_epochs = n_epochs
        self.compute_metrics()

        with trange(
            n_epochs, desc="training", file=sys.stdout, disable=False
        ) as pbar:
            # We have to use tqdm this way so it works in Jupyter notebook.
            # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
            for self.epoch in pbar:
                self.on_epoch_begin()
                pbar.update(1)
                for tensors_list in self.data_loaders_loop():
                    if tensors_list[0][0].shape[0] < 3:
                        continue
                    loss = self.loss(*tensors_list)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        self.model.eval()
        self.training_time += (time.time() - begin) - self.compute_metrics_time

    def loss(self, tensors):
        sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
        reconst_loss, kl_divergence = self.model(
            sample_batch, local_l_mean, local_l_var, batch_index
        )
        loss = torch.mean(reconst_loss + self.kl_weight * kl_divergence)
        return loss

    def on_epoch_begin(self):
        self.kl_weight = min(1, self.epoch / self.n_epochs_kl_warmup)

    @property
    def posteriors_loop(self):

        return ["train_set"]

    def data_loaders_loop(
        self
    ):  # returns an zipped iterable corresponding to loss signature

        data_loaders_loop = [self._posteriors[name] for name in self.posteriors_loop]
        return zip(
            data_loaders_loop[0],
            *[cycle(data_loader) for data_loader in data_loaders_loop[1:]]
        )

    def register_posterior(self, name, value):
        name = name.strip("_")
        self._posteriors[name] = value

    def __getattr__(self, name):
        if "_posteriors" in self.__dict__:
            _posteriors = self.__dict__["_posteriors"]
            if name.strip("_") in _posteriors:
                return _posteriors[name.strip("_")]
        return object.__getattribute__(self, name)


    def __setattr__(self, name, value):
        if isinstance(value, Posterior):
            name = name.strip("_")
            self.register_posterior(name, value)
        else:
            object.__setattr__(self, name, value)

    def train_test_validation(
        self,
        model=None,
        gene_dataset=None,
        train_size=0.1,
        test_size=None,
        type_class=Posterior,
    ):
        """Creates posteriors ``train_set``, ``test_set``, ``validation_set``.
            If ``train_size + test_size < 1`` then ``validation_set`` is non-empty.

            :param train_size: float, int, or None (default is 0.1)
            :param test_size: float, int, or None (default is None)
            """
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        n = len(gene_dataset)
        n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
        random_state = np.random.RandomState(seed=self.seed)
        permutation = random_state.permutation(n)
        indices_test = permutation[:n_test]
        indices_train = permutation[n_test : (n_test + n_train)]
        indices_validation = permutation[(n_test + n_train) :]

        return (
            self.create_posterior(
                model, gene_dataset, indices=indices_train, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, indices=indices_test, type_class=type_class
            ),
            self.create_posterior(
                model, gene_dataset, indices=indices_validation, type_class=type_class
            ),
        )

    def create_posterior(
        self,
        model=None,
        gene_dataset=None,
        shuffle=False,
        indices=None,
        type_class=Posterior,
    ):
        model = self.model if model is None and hasattr(self, "model") else model
        gene_dataset = (
            self.gene_dataset
            if gene_dataset is None and hasattr(self, "model")
            else gene_dataset
        )
        return type_class(
            model,
            gene_dataset,
            shuffle=shuffle,
            indices=indices,
            use_cuda=self.use_cuda,
            data_loader_kwargs=self.data_loader_kwargs,
        )



