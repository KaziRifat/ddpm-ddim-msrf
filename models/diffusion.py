"""
Diffusion Schedulers: DDPM & DDIM
Implements:
  - Forward process (q): adds noise according to variance schedule
  - DDPM reverse process (p): ancestral sampling with stochastic noise
  - DDIM reverse process: deterministic (eta=0) or interpolated (eta>0) sampling
  - Classifier-Free Guidance (CFG): merges conditional + unconditional predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


# ---------------------------------------------------------------------------
# Variance Schedule Utilities
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in 'Improved DDPM' (Nichol & Dhariwal 2021).
    More stable than linear schedule; avoids too-noisy images at the end.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Original DDPM linear schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


# ---------------------------------------------------------------------------
# Core Diffusion Class
# ---------------------------------------------------------------------------

class GaussianDiffusion:
    """
    Unified DDPM/DDIM diffusion process.

    Responsible for:
      1. Pre-computing all schedule-derived quantities (alphas, sigmas, etc.)
      2. Forward (noising) process: q(x_t | x_0)
      3. DDPM reverse process: p(x_{t-1} | x_t, x_0_pred)
      4. DDIM reverse process: deterministic or stochastic
      5. Computing training loss (simple L_simple or hybrid)
      6. Classifier-Free Guidance support
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = 'cosine',
        loss_type: str = 'l2',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Build beta schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == 'quadratic':
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # Pre-compute derived quantities — store all as float32 tensors
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1).sqrt()

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Gather tensor values at timesteps t and broadcast to `shape`."""
        B = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(B, *((1,) * (len(shape) - 1)))

    # -----------------------------------------------------------------------
    # Forward Process
    # -----------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) using the reparameterization trick.
        x_t = sqrt(ā_t) * x_0 + sqrt(1 - ā_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    # -----------------------------------------------------------------------
    # Posterior
    # -----------------------------------------------------------------------

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance q(x_{t-1} | x_t, x_0)."""
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    # -----------------------------------------------------------------------
    # Predict x_0 from noise prediction
    # -----------------------------------------------------------------------

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Invert q_sample: given x_t and predicted noise, recover x_0."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # -----------------------------------------------------------------------
    # DDPM Reverse Step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        One DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t).
        Optionally applies Classifier-Free Guidance (CFG).
        """
        pred_noise = self._get_guided_noise(model, x_t, t, class_labels, cfg_scale, null_class)

        x_start = self.predict_start_from_noise(x_t, t, pred_noise).clamp(-1, 1)
        mean, _, log_var = self.q_posterior(x_start, x_t, t)

        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero = (t > 0).float().reshape(-1, 1, 1, 1)
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape: Tuple,
        device: torch.device,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Full DDPM ancestral sampling loop."""
        x = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="DDPM Sampling")
            except ImportError:
                pass

        for i in iterator:
            t = torch.full((shape[0],), i, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t, class_labels, cfg_scale, null_class)

        return x

    # -----------------------------------------------------------------------
    # DDIM Reverse Step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        One DDIM reverse step.
        eta=0  → fully deterministic (DDIM)
        eta=1  → recovers DDPM
        0<eta<1→ interpolation
        """
        pred_noise = self._get_guided_noise(model, x_t, t, class_labels, cfg_scale, null_class)

        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = self._extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t.shape)

        # Predicted x_0
        x0_pred = (x_t - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Compute sigma for stochastic step
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()

        # Direction pointing to x_t
        dir_xt = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise

        noise = torch.randn_like(x_t) if eta > 0 else 0
        return alpha_prev.sqrt() * x0_pred + dir_xt + sigma * noise

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape: Tuple,
        device: torch.device,
        ddim_steps: int = 50,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        null_class: Optional[int] = None,
        eta: float = 0.0,
        progress: bool = True,
    ) -> torch.Tensor:
        """Full DDIM sampling loop with arbitrary number of steps."""
        x = torch.randn(shape, device=device)

        # Subsample timesteps uniformly
        step_indices = np.linspace(0, self.timesteps - 1, ddim_steps + 1, dtype=int)
        timesteps = list(reversed(step_indices[1:]))    # T -> 0
        prev_timesteps = list(reversed(step_indices[:-1]))

        iterator = zip(timesteps, prev_timesteps)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc=f"DDIM Sampling ({ddim_steps} steps)")
            except ImportError:
                pass

        for t_val, t_prev_val in iterator:
            t = torch.full((shape[0],), t_val, dtype=torch.long, device=device)
            t_prev = torch.full((shape[0],), t_prev_val, dtype=torch.long, device=device)
            x = self.ddim_sample(model, x, t, t_prev, class_labels, cfg_scale, null_class, eta)

        return x

    # -----------------------------------------------------------------------
    # Classifier-Free Guidance
    # -----------------------------------------------------------------------

    def _get_guided_noise(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        cfg_scale: float,
        null_class: Optional[int],
    ) -> torch.Tensor:
        """
        Classifier-Free Guidance (Ho et al. 2022):
        ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
        
        Requires model trained with null class token (null_class index).
        """
        if cfg_scale == 1.0 or class_labels is None:
            return model(x_t, t, class_labels)

        # Conditional prediction
        eps_cond = model(x_t, t, class_labels)

        # Unconditional prediction (null class)
        null_labels = torch.full_like(class_labels, null_class)
        eps_uncond = model(x_t, t, null_labels)

        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    # -----------------------------------------------------------------------
    # Training Loss
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        cfg_dropout_prob: float = 0.1,
        null_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute training loss L_simple = E[||ε - ε_θ(x_t, t)||²].

        CFG training: randomly drops class label with probability `cfg_dropout_prob`
        by replacing it with `null_class`. This forces the model to learn both
        conditional and unconditional distributions.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)

        # CFG label dropout during training
        if class_labels is not None and null_class is not None and cfg_dropout_prob > 0:
            mask = torch.rand(x_start.shape[0], device=x_start.device) < cfg_dropout_prob
            class_labels = class_labels.clone()
            class_labels[mask] = null_class

        pred_noise = model(x_noisy, t, class_labels)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_noise, noise)
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(pred_noise, noise)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    # -----------------------------------------------------------------------
    # Utility: SNR weighting (optional, improves training stability)
    # -----------------------------------------------------------------------

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio at timestep t."""
        alphas = self._extract(self.alphas_cumprod, t, t.shape)
        return alphas / (1 - alphas)

    def min_snr_weight(self, t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
        """Min-SNR-γ loss weighting (Hang et al. 2023)."""
        snr_val = self.snr(t)
        return torch.minimum(snr_val, torch.full_like(snr_val, gamma)) / snr_val


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    diffusion = GaussianDiffusion(timesteps=1000, schedule='cosine')

    # Verify forward process
    x0 = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4,))
    noise = torch.randn_like(x0)
    xt = diffusion.q_sample(x0, t, noise)
    print(f"q_sample output shape: {xt.shape}")

    # Verify schedule at t=0 and t=999
    t0 = torch.tensor([0])
    tT = torch.tensor([999])
    print(f"sqrt_alpha_cumprod at t=0:   {diffusion.sqrt_alphas_cumprod[0]:.4f}  (should be ~1)")
    print(f"sqrt_alpha_cumprod at t=999: {diffusion.sqrt_alphas_cumprod[-1]:.4f} (should be ~0)")
    print(f"sqrt_one_minus at t=0:   {diffusion.sqrt_one_minus_alphas_cumprod[0]:.4f}  (should be ~0)")
    print(f"sqrt_one_minus at t=999: {diffusion.sqrt_one_minus_alphas_cumprod[-1]:.4f} (should be ~1)")
    print("Diffusion schedule OK ✓")
