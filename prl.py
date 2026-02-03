import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class PMLoss(torch.nn.Module):
    def __init__(self, K=20.0, p=2, sigma=1.0, rho=0.5):
        super(PMLoss, self).__init__()
        self.K = K
        self.p = p
        self.sigma = sigma
        self.rho = rho

        # The Sobel Operator
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1],
                                                      [-2, 0, 2],
                                                      [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[1, 2, 1],
                                                      [0, 0, 0],
                                                      [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3))

    def gaussian_kernel(self, sigma, device):
        size = int(2 * ceil(3 * sigma) + 1) if sigma > 0 else 3
        size = size if size % 2 != 0 else size + 1
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def get_div(self, img, g_map):
        # Eq.15 and Eq.16
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)

        flux_x = g_map * grad_x
        flux_y = g_map * grad_y

        # Calculate the diverse of X and Y
        div_x = F.conv2d(flux_x, self.sobel_x, padding=1)
        div_y = F.conv2d(flux_y, self.sobel_y, padding=1)
        return div_x + div_y

    def edge_stop(self, val):
        return 1.0 / (1.0 + (torch.abs(val) / self.K) ** self.p)

    def compute_pde_term(self, I):
        device = I.device
        # Eq. 7
        kernel_sigma = self.gaussian_kernel(self.sigma, device)
        I_sigma = F.conv2d(I, kernel_sigma, padding=kernel_sigma.shape[-1] // 2)

        # The gradient of I_sigma
        I_s_x = F.conv2d(I_sigma, self.sobel_x, padding=1)
        I_s_y = F.conv2d(I_sigma, self.sobel_y, padding=1)
        grad_norm = torch.sqrt(I_s_x ** 2 + I_s_y ** 2 + 1e-6)

        # Eq. 8
        J11 = I_s_x * I_s_x
        J22 = I_s_y * I_s_y
        J12 = I_s_x * I_s_y

        # Eq. 9
        kernel_rho = self.gaussian_kernel(self.rho, device)
        pad_r = kernel_rho.shape[-1] // 2
        J11_rho = F.conv2d(J11, kernel_rho, padding=pad_r)
        J22_rho = F.conv2d(J22, kernel_rho, padding=pad_r)
        J12_rho = F.conv2d(J12, kernel_rho, padding=pad_r)

        # Eq. 11
        g_for_I = self.edge_stop(torch.sqrt(J11_rho + J22_rho + 1e-6))
        g_for_J = self.edge_stop(grad_norm)

        # The right of Eq. 11
        div_I = self.get_div(I, g_for_I)
        div_J = self.get_div(J11_rho, g_for_J) + self.get_div(J22_rho, g_for_J)

        return div_I, div_J

    def forward(self, pred, target):
        div_I_pred, div_J_pred = self.compute_pde_term(pred)
        div_I_trgt, div_J_trgt = self.compute_pde_term(target)

        # Eq.12
        loss_I = F.mse_loss(div_I_pred, div_I_trgt)
        loss_J = F.mse_loss(div_J_pred, div_J_trgt)

        return loss_I + loss_J