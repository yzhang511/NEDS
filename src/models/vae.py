import torch
import torch.nn as nn
import numpy as np

class GLVM_VAE(nn.Module):
    """ " Gaussian VAE example
    Encoding: p(z|x,m)
    Sampling: reparametrise
    Decoder: p(x|z)

    the networks should know which data is actually missing and which one is not

    combination indicates if the mask embedding should be concatenated or multiplied
    """

    def __init__(
        self,
        input_size,
        latent_size,
        args,
        nonlin=nn.Identity(),
        impute_missing=False,
        init="norm",
        scale=0.1,
        uncertainty=False,
        combination="regular",
        n_hidden=10,
        freeze_gen=False,  # freeze the generative model
        C=None,
        d=None,
        noise_std=None,
        n_masks=2,
        dropout=0.0,
    ):
        print("GLVM_VAE")
        super().__init__()

        assert type(latent_size) == int
        assert type(input_size) == int

        self.latent_size = latent_size
        self.added_std = args.added_std

        # should mask be passed to network
        self.pass_mask = args.pass_mask
        self.pass_mask_decoder = (
            args.pass_mask_decoder
        )  # also to decoder default same as pass_mask

        # uncertainty output predictions learned
        self.uncertainty = uncertainty

        # learned imputation
        self.impute_missing = impute_missing
        self.init = init
        self.scale = scale
        self.impute = ImputeMissing(
            input_size=input_size, init=self.init, scale=self.scale
        )

        # init encoder and decoder
        self.encoder = GLVM_Encoder(
            input_size=input_size,
            latent_size=latent_size,
            masked=self.pass_mask,
            nonlin=nonlin,
            combination=combination,
            n_hidden=n_hidden,
            n_masks=n_masks,
            dropout=dropout,
        )
        self.decoder = GLVM_Decoder(
            input_size=input_size,
            latent_size=latent_size,
            masked=self.pass_mask_decoder,
            nonlin=nonlin,
            uncertainty=self.uncertainty,
            n_hidden=n_hidden,
            freeze_gen=freeze_gen,
            C=C,
            d=d,
            noise_std=noise_std,
        )

    def forward(self, x, m=None):

        if self.impute_missing:
            x = self.impute(x, m)
        # encode
        means, log_var = self.encoder(x, m)
        # reparametrization trick
        z = self.reparameterize(means, log_var)
        # decode
        recon_mean, recon_var = self.decoder(z, m)

        return recon_mean, recon_var, means, log_var, z, None

    def reparameterize(self, mu, log_var):
        """reparameterization trick with added value to std to ensure
        posterior variance is never below added_std^2 often 1e-3 taken"""

        std = torch.exp(0.5 * log_var) + self.added_std
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z, m=None):
        """Decode step of the VAE p(x|z)"""

        recon_mean, recon_var = self.decoder(z, m)

        return recon_mean, recon_var

    def inference(self, x, m=None):
        """Inference step of the VAE p(z|x,m)"""
        if self.impute_missing:
            x = self.impute(x, m)

        means, log_var = self.encoder(x, m)
        return means, log_var


class GLVM_Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        masked,
        nonlin=nn.ReLU(),
        combination="concat",
        n_hidden=10,
        n_masks=2,
        dropout=0.0,
    ):
        super().__init__()
        self.pass_mask = masked

        # set up encoder modules
        self.input_size = input_size
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.nonlin = nonlin
        self.n_masks = n_masks

        # dropout contrast to the decoder
        self.dropout_val = dropout
        self.dropout = nn.Dropout(self.dropout_val)

        self.combination = combination
        # implement different methods to combine mask and input
        if self.combination == "concat":
            # concatenation doubles the input size
            self.input_size_concat = int(2 * self.input_size)
        elif self.combination == "multiply" or self.combination == "add":
            # adding and multiplying keep the input size
            self.input_size_concat = self.input_size
        else:
            # setup regular encoder modules
            self.mask_embed_size = self.input_size
            # add mask embedding size to input size
            self.input_size_concat = self.input_size + self.mask_embed_size
            # specify learneable linear mask embedding
            self.mask_embdeding = nn.Linear(self.input_size, self.mask_embed_size)
            # first expand the dimensions linearly
            self.relu_hidden_dim_expansion = nn.Sequential(
                nn.Linear(self.input_size_concat, self.n_hidden), nn.Identity()
            )
            # then reduce the dimensions again non linearly
            self.relu_hidden_dim_reduction = nn.Sequential(
                nn.Linear(self.n_hidden, self.input_size_concat),
                self.nonlin,
                nn.Linear(self.input_size_concat, self.input_size_concat),
                self.nonlin,
            )
            # perform linear mapping to mean and non linear to log var
            self.linear_means = nn.Linear(self.input_size_concat, self.latent_size)
            self.linear_log_var = nn.Sequential(
                nn.Linear(self.input_size_concat, 10),
                self.nonlin,
                nn.Linear(10, self.latent_size),
            )

        if self.combination in ["concat", "multiply", "add"]:

            # additional modules for advanced combination of mask and input
            self.mask_embed_hidden = int(np.round(self.input_size / 2))
            # mask embedding learneable
            self.mask_embed_advanced = nn.Sequential(
                nn.Linear(self.input_size, self.mask_embed_hidden),
                nn.Linear(self.mask_embed_hidden, self.mask_embed_hidden),
                self.nonlin,
                nn.Linear(self.mask_embed_hidden, self.input_size),
            )
            #  input embedding
            self.input_embed_advanced = nn.Sequential(
                nn.Linear(self.input_size, self.n_hidden),
                nn.Linear(self.n_hidden, self.n_hidden),
                self.nonlin,
                nn.Linear(self.n_hidden, self.input_size),
            )

            # linear mapping before the combination with the mask
            self.linear_mean_branch = nn.Linear(self.input_size, self.input_size)
            # different linear mapping for the posterior variance to allow for cutting out x - effects
            self.linear_log_var_branch = nn.Linear(
                self.input_size, self.input_size
            )  # here no softmax as logvae not sigma

            self.mean_branch_transform = nn.Sequential(
                nn.Linear(self.input_size_concat, self.mask_embed_hidden),
                nn.Linear(self.mask_embed_hidden, self.mask_embed_hidden),
                self.nonlin,
                nn.Linear(self.mask_embed_hidden, self.input_size_concat),
            )

            # perform linear read out
            self.linear_means = nn.Linear(self.input_size_concat, self.latent_size)
            self.linear_log_var = nn.Linear(
                self.input_size_concat, self.latent_size
            )  # here no softmax as logvae not sigma

    def forward(self, x, m=None):
        """Computes the forward pass through the Encoder.
        different methods to combine mask and input are supported"""

        if not self.pass_mask:
            # keep the dimensions the same _> also without mask pass 0 -> no information
            m = torch.ones_like(x)

        if self.combination in ["concat", "multiply", "add"]:
            # alternative methods to combine mask and input

            # embed the mask
            m = self.mask_embed_advanced(m)
            # embed input
            x = self.input_embed_advanced(x)

            mx = self.linear_mean_branch(x)
            varx = self.linear_log_var_branch(x)

            # concatenate the mask and the input or multiply
            if self.combination == "concat":
                mx = torch.cat((mx, m), dim=1)
                varx = torch.cat((varx, m), dim=1)
            elif self.combination == "multiply":
                mx = mx * m
                varx = varx * m
            elif self.combination == "add":
                mx = mx + m
                varx = varx + m

            mx = self.mean_branch_transform(mx)
            # map to mean and log var

            means = self.linear_means(mx)
            log_vars = self.linear_log_var(varx)
        else:  # regular encoder method
            # embed the mask first that it can
            m = self.mask_embdeding(m)

            # dropping out in the input layer
            if self.dropout_val != 0.0:
                x = self.dropout(x)
            # concatenate the mask and the input
            x = torch.cat((x, m), dim=1)

            # MLP dim expansion and reduction
            x = self.relu_hidden_dim_expansion(x)
            x = self.relu_hidden_dim_reduction(x)

            # map to mean and log var
            means = self.linear_means(x)
            log_vars = self.linear_log_var(x)

        return means, log_vars


class GLVM_Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        masked,
        nonlin=nn.ReLU(),
        uncertainty=False,
        n_hidden=10,
        freeze_gen=False,
        C=None,
        d=None,
        noise_std=None,
    ):
        super().__init__()

        if (
            freeze_gen
        ):  # if we freeze the generative model ensure the parameters are passed
            assert C is not None, "Provide C for Cz + d"
            assert d is not None, "Provide d for Cz + d"
            if (
                uncertainty
            ):  # if we also estimate the predictive variance ensure the observation noise term is passed
                assert noise_std is not None, "Provide noise_std"

        # layer specifications
        self.pass_mask = masked
        self.input_size = input_size
        self.uncertainty = uncertainty
        self.mask_embed_size = int(np.round(input_size / 2))
        self.input_size_concat = self.input_size + self.mask_embed_size
        self.n_hidden = n_hidden
        self.nonlin = nonlin

        self.freeze_gen = freeze_gen
        if freeze_gen:
            self.C = C
            self.d = d
            self.gt_variance = noise_std * noise_std
        else:
            # mask embedding
            self.mask_embdeding = nn.Linear(self.input_size, self.mask_embed_size)
            # first expand the dimensions linearly
            self.relu_hidden_dim_expansion = nn.Sequential(
                nn.Linear(self.input_size_concat, self.n_hidden), nn.Identity()
            )
            # then reduce the dimensions again non linearly
            self.relu_hidden_dim_reduction = nn.Sequential(
                nn.Linear(self.n_hidden, self.input_size_concat),
                self.nonlin,
                nn.Linear(self.input_size_concat, self.input_size_concat),
                self.nonlin,
            )

            # set up decoder modules
            self.dim_increase = nn.Linear(latent_size, self.input_size)
            # output mean and variance to dataspace softplus to ensure positive noise term
            self.mean_linear = nn.Linear(self.input_size_concat, self.input_size)
            self.var_linear = nn.Sequential(
                nn.Linear(self.input_size_concat, self.input_size), nn.Softplus()
            )

    def forward(self, z, m):

        if self.freeze_gen:
            mean = self.C.float() @ z.T + self.d

            # if we estimate the predictive variance
            if self.uncertainty:
                # set the variance to the ground truth variance
                var = self.gt_variance
                var = var * torch.ones_like(mean)
                return mean.T, var.T
            else:
                return mean.T, None
        else:
            # expand the dimensions
            x = self.dim_increase(z)
            # add in mask after dim increase even if mask not masked just pass zeros
            # in most cases the decoder mask is not passed both for naive and masked conditions
            if not self.pass_mask:
                m = torch.ones_like(x)
            m = self.mask_embdeding(m)

            # concatenate the mask and the input
            x = torch.cat((x, m), dim=1)
            x = self.relu_hidden_dim_expansion(x)
            # expand the dimensions
            x = self.relu_hidden_dim_reduction(x)
            #  linear layer to output
            mean = self.mean_linear(x)
            if self.uncertainty:
                var = self.var_linear(x)
                return mean, var
            else:
                return mean, None


class ImputeMissing(nn.Module):
    def __init__(self, input_size, init="norm", scale=0.1):

        super().__init__()
        # set up encoder modules
        self.input_size = input_size
        # ini
        # self.alpha = nn.Parameter(torch.ones(input_size, requires_grad=True))
        if init == "one":
            self.alpha = nn.Parameter(
                torch.ones(input_size, requires_grad=True).float()
            )
        elif init == "zero":
            self.alpha = nn.Parameter(
                torch.zeros(input_size, requires_grad=True).float()
            )
        elif init == "uni":
            self.alpha = nn.Parameter(
                torch.tensor(
                    scale * np.random.rand(input_size), requires_grad=True
                ).float()
            )
        else:
            self.alpha = nn.Parameter(
                torch.tensor(
                    np.random.normal(0, scale, size=(input_size)), requires_grad=True
                ).float()
            )

    def forward(self, x, m=None):
        """Imputes missing values with learned parameter alpha

        Args:
          x: input vector
          m: mask

        Returns:
          modified x with imputed alpha values instead of 0s for missing values
        """
        x = x + (1 - m) * self.alpha

        return x
