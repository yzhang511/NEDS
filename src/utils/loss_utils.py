import torch


def masked_GNLL_loss_fn(
    recon_x,
    recon_var,
    x,
    mean,
    log_var,
    mask,
    eps=1e-06,
    beta=1,
    full=False,
):
    """Calculate the Guassian negative log likelihood for a Gaussian with mean recon_x and variance recon_var
    Consider only observed (not masked) values unless explicitly stated otherwise with full=True

    Parameters
    ----------
        x : tensor with ground truth to both estimate the sigma in optimal-sigma-vae and calc masked and unmasked loss
        recon_x : reconstructed mean
        recon_var : tensor predicted variances decoder
        mean: tensor with predicted approx posterior means
        log_var: log variance of the approx posterior
        mask: mask tensor of 0s and 1s
        eps: epsilon for numerical stability
        beta: beta for KL divergence
        full: bool for full loss masked and observed
    Returns
        total loss GNLL, unobserved loss, KL divergence, unobserved loss, mean variance

    Note: Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
    ensures stable training. eps in Gaussian NLL ensures that training is stable np.exp(-6)**2 roughly 6e-6 so std. implemented one is sufficient"""

    batch_size = x.size(0)

    loss = torch.nn.GaussianNLLLoss(eps=eps, reduction="none")

    GNLL_full = loss(recon_x, x, recon_var)

    # observed reconstruction loss mask 1 for observed 0 for unobserved
    GNLL = mask * GNLL_full
    GNLL = torch.sum(GNLL)

    # unobserved reconstruction loss
    GNLL_unobs = (1 - mask) * GNLL_full  # not mask -> unobserved
    GNLL_unobs = torch.sum(GNLL_unobs)

    # KL divergence between approximate posterior and standard Gaussian N(0,1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if full:
        # use the entire loss masked and observed
        return (
            (torch.sum(GNLL_full) + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )
    else:
        return (
            (GNLL + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )


def masked_MSE_loss_fn(
    recon_x,
    recon_var,
    x,
    mean,
    log_var,
    mask,
    eps=1e-06,
    beta=1,
    full=False,
):
    """Calculate the Guassian negative log likelihood for a Gaussian with mean recon_x and variance recon_var
    Consider only observed (not masked) values unless explicitly stated otherwise with full=True

    Parameters
    ----------
        x : tensor with ground truth to both estimate the sigma in optimal-sigma-vae and calc masked and unmasked loss
        recon_x : reconstructed mean
        recon_var : tensor predicted variances decoder
        mean: tensor with predicted approx posterior means
        log_var: log variance of the approx posterior
        mask: mask tensor of 0s and 1s
        eps: epsilon for numerical stability
        beta: beta for KL divergence
        full: bool for full loss masked and observed
    Returns
        total loss GNLL, unobserved loss, KL divergence, unobserved loss, mean variance

    Note: Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
    ensures stable training. eps in Gaussian NLL ensures that training is stable np.exp(-6)**2 roughly 6e-6 so std. implemented one is sufficient"""

    batch_size = x.size(0)

    GNLL_full = torch.nn.functional.mse_loss(recon_x, x, reduction="none")

    # observed reconstruction loss mask 1 for observed 0 for unobserved
    GNLL = mask * GNLL_full
    GNLL = torch.sum(GNLL)

    # unobserved reconstruction loss
    GNLL_unobs = (1 - mask) * GNLL_full  # not mask -> unobserved
    GNLL_unobs = torch.sum(GNLL_unobs)

    # KL divergence between approximate posterior and standard Gaussian N(0,1)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if full:
        # use the entire loss masked and observed
        return (
            (torch.sum(GNLL_full) + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )
    else:
        return (
            (GNLL + beta * KLD) / batch_size,
            GNLL / batch_size,
            KLD / batch_size,
            GNLL_unobs / batch_size,
            torch.mean(recon_var, dim=0),
        )
