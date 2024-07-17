import torch
import torch.nn as nn
import numpy as np
from models.vae import GLVM_VAE
from models.vae_masks import MultipleDimGauss
from utils.loss_utils import masked_GNLL_loss_fn, masked_MSE_loss_fn

import copy
from copy import deepcopy

class ModelGLVM(object):
    def __init__(
        self,
        args,
        logs,
        inference_model=GLVM_VAE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        Generator=MultipleDimGauss,
        nonlin=nn.ReLU(),
        dropout=0.0,
    ):

        """
        Initialize with a single dict that contains the parameters for the network.
        the chosen dataset, the infernece network model and the training device
        All other parameters for training and evaluation can be changed manually after initialization.

        Parameters
        ----------
        args: dict
            All parameters that should be changed / added to the default attributes of the network class.

        args: inference_model
            Inference network with stanard VAE as default

        args: device
            cpu if no gpu detected

        """
        self.args = args
        self.device = device
        self.dropout = dropout
        print(self.args)

        print("Run model")

        # beta parameter for the KL term
        self.beta = args.beta
        self.beta_copy = copy.deepcopy(self.beta)
        self.args.betastep = self.beta / self.args.warmup_range

        # logging of error metrics elbo, kl and reconstruction loss,
        self.logs = logs

        # select current method
        self.method = args.method

        # Imputation of masked values (zero, mean, random, val)
        self.one_impute = self.args.one_impute
        self.mean_impute = self.args.mean_impute and not self.one_impute
        self.val_impute = self.args.val_impute and not (
            self.one_impute or self.mean_impute
        )
        self.random_impute = self.args.random_impute and not (
            self.one_impute or self.mean_impute or self.val_impute
        )

        self.x_dim = self.args.x_dim
        self.n_samples = self.args.n_samples
        self.nonlin = nonlin

        # specify the loss gnll vs mse
        # self.loss_fn = masked_gaussian_logprob_loss_fn if self.args.uncertainty else masked_mse_loss_fn
        self.loss_fn = (
            masked_GNLL_loss_fn if self.args.uncertainty else masked_MSE_loss_fn
        )
        print(
            "Gaussian Neg. log lik loss"
            if self.args.uncertainty
            else "Standard MSE loss"
        )

        # Here we compare three methods zero imputation

        # methods to be compared
        # 1. just set unobserved values to zero and don't provide info which point is missing which is a zero data val
        # 2. set unobserved values to zero and shift observed values by a fixed baseline (zero data val no longer exists)
        # 3. set unobserved values to zero but concatenate the mask with the input image in the decoder

        if self.method == "zero_imputation_baselined":
            self.baselined = True  # add baseline to all observed
            self.args.pass_mask = False  # pass mask to the encoder and decoder
            self.args.pass_mask_decoder = self.args.pass_mask
            self.impute_missing = False

        if self.method == "zero_imputation":
            self.args.pass_mask = False
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = False
            self.impute_missing = False

        if self.method == "zero_imputation_mask_concatenated":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = False
            self.impute_missing = True

        if self.method == "zero_imputation_mask_concatenated_encoder_only":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = False
            self.baselined = False
            self.impute_missing = True

        if self.method == "zero_imputation_mask_concatenated_baselined":
            self.args.pass_mask = True
            self.args.pass_mask_decoder = self.args.pass_mask
            self.baselined = True
            self.impute_missing = True

        if args.imputeoff:
            self.impute_missing = False

        self.baseline = self.args.baseline

        self.masked = self.args.masked
        np.random.seed(1)  # ensure always the same mask is chosen
        self.mask_generator = Generator(
            self.x_dim, self.args.n_masked_vals, n_masks=self.args.unique_masks
        )
        np.random.seed(self.args.seed)
        self.n_unique_masks = self.mask_generator.n_unique_masks

        self.vae = inference_model(
            input_size=self.x_dim,
            latent_size=self.args.latent_size,
            args=self.args,
            impute_missing=self.impute_missing,
            nonlin=self.nonlin,
            uncertainty=self.args.uncertainty,
            combination=self.args.combination,
            n_hidden=self.args.n_hidden,
            freeze_gen=self.args.freeze_decoder,
            C=torch.tensor(self.args.C).to(device),
            d=torch.tensor(self.args.d).to(device),
            noise_std=torch.tensor(self.args.noise).to(device),
            n_masks=self.n_unique_masks + 1,
            dropout=self.dropout,
        ).to(device)

        # set up optimizer
        self.optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=self.args.learning_rate
        )

        # set up time stamp
        self.ts = args.ts if args.ts is not None else time.time()

        # save amount of trainable parameters
        print("Traineable parameters: ", self.count_parameters(self.vae))
        self.args.n_parameters = self.count_parameters(self.vae)
        args.n_parameters = self.args.n_parameters

        # set up model path
        model_dir_path = os.path.join(self.args.fig_root, str(self.ts), self.method)
        ensure_directory(model_dir_path)