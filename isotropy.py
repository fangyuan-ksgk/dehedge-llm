"""Isotropy regularization — the de-hedging loss that worked best across all our experiments.

Representation hedging: a backprop-trained LM spends its dominant readout directions on the *marginal* (frequent
tokens) and starves the rare ones. Two cheap, training-only fixes (used together = "de-hedging"):

  1. marginal BIAS CHANNEL  -> see `log_unigram_bias` (init an additive logit bias to the log-unigram so the model
     does not have to encode the unconditional marginal inside the representation).
  2. ISOTROPY regularizer    -> `iso_loss` below (VICReg-style: variance-floor + decorrelation), applied to the final
     hidden state. It recruits the unused directions and decorrelates them, so capacity is spread instead of hedged.

`iso_loss` is the variant that beat alternatives in our sweeps (SIGReg / full-Gaussian, head-only whitening, etc.):
a variance *floor* (>=1) rather than a hard unit-variance cap, plus off-diagonal covariance decorrelation.
"""
import torch


def iso_loss(rep: torch.Tensor) -> torch.Tensor:
    """VICReg-style isotropy on a representation.

    Args:
        rep: (..., D) hidden states (any leading shape; flattened over tokens).
    Returns:
        scalar loss = variance-floor term + decorrelation term.
            variance-floor: hinge pushing every coordinate's std up to >= 1  (recruit dormant dims)
            decorrelation : squared off-diagonal covariance, normalized by D  (spread, don't duplicate)
    Add to the task loss as:  total = ce + lambda_iso * iso_loss(rep)   (lambda_iso ~ 0.10 worked best).
    """
    z = rep.reshape(-1, rep.size(-1))
    z = z - z.mean(0)
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    var = torch.diagonal(cov)
    v_term = torch.relu(1.0 - torch.sqrt(var + 1e-6)).mean()
    c_term = (cov - torch.diag(var)).pow(2).sum() / d
    return v_term + c_term


def log_unigram_bias(token_counts: torch.Tensor) -> torch.Tensor:
    """Init for the marginal bias channel: centered log-unigram over the vocabulary.

    Args:
        token_counts: (V,) integer/float token counts from the training corpus.
    Returns:
        (V,) centered log( count + 1 ) — use to initialize an additive output-logit bias parameter.
    """
    lu = torch.log(token_counts.float() + 1.0)
    return lu - lu.mean()
