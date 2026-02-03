# Physical Regularization Loss (PRL) for Image Segmentation

PyTorch implementation of the **Physical Regularization Loss (PRL)**, as described in the paper: *"Physical Regularization Loss: Integrating Physical Knowledge to Image Segmentation"*.

This loss function integrates physical constraints via anisotropic diffusion equations (Perona-Malik) into the deep learning optimization process to enhance edge preservation and model generalization.

---

## 📖 Overview

Standard data-driven segmentation models (like U-Net or DeepLab) often struggle with blurry boundaries or limited training data. **PRL** addresses this by:
- **Incorporating Physical Laws**: Modeling the segmentation process as a heat diffusion evolution.
- **Anisotropic Smoothing**: Applying high diffusion in homogeneous regions while stopping at object boundaries using a **Structure Tensor**.
- **Edge-Preserving Regularization**: Penalizing the model when its prediction's physical divergence deviates from the ground truth's geometric structure.

---

## 🛠️ Implementation (`loss.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class PMLoss(nn.Module):
    """
    Physical Regularization Loss (PRL) Implementation.
    Ref: Physical Regularization Loss: Integrating Physical Knowledge to Image Segmentation
    """
    def __init__(self, K=20.0, p=2, sigma=1.0, rho=0.5):
        """
        Args:
            K (float): Edge-sensitive threshold (K=20.0 recommended for 0-255 scale).
            p (int): Exponent controlling the diffusion nonlinearity (p=2 recommended).
            sigma (float): Std for initial image smoothing (I_sigma).
            rho (float): Std for structural tensor smoothing (J_rho).
        """
        super(PMLoss, self).__init__()
        self.K = K
        self.p = p
        self.sigma = sigma
        self.rho = rho
        
        # Sobel operators for gradient computation (Paper Eq. 18)
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                     [-2, 0, 2],
                                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([[1, 2, 1],
                                                     [0, 0, 0],
                                                     [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3))

    def _gaussian_kernel(self, sigma, device):
        """Generates a 2D Gaussian kernel for smoothing."""
        size = int(2 * ceil(3 * sigma) + 1)
        size = size if size % 2 != 0 else size + 1
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _get_divergence(self, img, g_map):
        """Calculates div(g * grad(I)) using half-point finite difference."""
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        
        # Flux calculation
        flux_x = g_map * grad_x
        flux_y = g_map * grad_y
        
        # Divergence calculation using numerical differentiation
        div_x = F.conv2d(flux_x, self.sobel_x, padding=1)
        div_y = F.conv2d(flux_y, self.sobel_y, padding=1)
        return div_x + div_y

    def _edge_stop_function(self, val):
        """Edge stop function g(x) = 1 / (1 + (|x|/K)^p)."""
        return 1.0 / (1.0 + (torch.abs(val) / self.K) ** self.p)

    def _compute_pde_components(self, I):
        """Computes the physical divergence terms for the coupled system."""
        device = I.device
        
        # 1. Smooth image to get I_sigma (Eq. 7)
        k_sigma = self._gaussian_kernel(self.sigma, device)
        I_sigma = F.conv2d(I, k_sigma, padding=k_sigma.shape[-1]//2)
        
        # 2. Get gradients for Structure Tensor
        I_s_x = F.conv2d(I_sigma, self.sobel_x, padding=1)
        I_s_y = F.conv2d(I_sigma, self.sobel_y, padding=1)
        grad_norm = torch.sqrt(I_s_x**2 + I_s_y**2 + 1e-6)
        
        # 3. Structure Tensor components J = grad(I_sigma) * grad(I_sigma)^T
        J11 = I_s_x * I_s_x
        J22 = I_s_y * I_s_y
        
        # 4. Smooth Structure Tensor to get J_rho (Eq. 9)
        k_rho = self._gaussian_kernel(self.rho, device)
        pad_r = k_rho.shape[-1]//2
        J11_rho = F.conv2d(J11, k_rho, padding=pad_r)
        J22_rho = F.conv2d(J22, k_rho, padding=pad_r)
        
        # 5. Calculate diffusion coefficients (Eq. 11)
        g_for_I = self._edge_stop_function(torch.sqrt(J11_rho + J22_rho + 1e-6))
        g_for_J = self._edge_stop_function(grad_norm)
        
        # 6. Final Divergence terms
        div_I = self._get_divergence(I, g_for_I)
        div_J = self._get_divergence(J11_rho, g_for_J) + self._get_divergence(J22_rho, g_for_J)
        
        return div_I, div_J

    def forward(self, pred, target):
        """
        Args:
            pred: Prediction probability map [B, 1, H, W]
            target: Ground truth binary mask [B, 1, H, W]
        """
        div_I_p, div_J_p = self._compute_pde_components(pred)
        div_I_t, div_J_t = self._compute_pde_components(target)
        
        loss_I = F.mse_loss(div_I_p, div_I_t)
        loss_J = F.mse_loss(div_J_p, div_J_t)
        
        return loss_I + loss_J
