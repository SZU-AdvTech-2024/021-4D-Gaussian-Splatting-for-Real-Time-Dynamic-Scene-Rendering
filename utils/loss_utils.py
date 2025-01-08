#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()
# def l1_loss(network_output, gt, beta=0.168, alpha=0.52, gamma=1.0):
#     """
#     Modified Smooth L1 Loss function.

#     Parameters:
#     - network_output: Tensor, predicted values.
#     - gt: Tensor, target values.
#     - beta: float, controls the point where the loss changes from L2 to L1.
#     - alpha: float, weight for the L2 region (small differences).
#     - gamma: float, weight for the L1 region (large differences).

#     Returns:
#     - loss: Tensor, modified smooth L1 loss.
#     """
#     diff = torch.abs(network_output - gt)
    
#     # Modified formula with weighted L2 and L1 regions
#     loss = torch.where(
#         diff < beta,
#         alpha * (diff ** 2) / (2 * beta),  # L2 region with weight alpha
#         gamma * (diff - 0.5 * beta)       # L1 region with weight gamma
#     )
    
#     return loss.mean()
# def l1_loss(network_output, gt):
#     """
#     Smooth L1 Loss function.

#     Parameters:
#     - network_output: Tensor, predicted values.
#     - gt: Tensor, target values.

#     Returns:
#     - loss: Tensor, smooth L1 loss.
#     """
#     diff = network_output - gt
#     abs_diff = torch.abs(diff)

#     # Smooth L1 loss calculation
#     loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)

#     return loss.mean()
    """
    Adaptive Smooth L1 Loss function with dynamic weights and improved stability.

    Parameters:
    - network_output: Tensor, predicted values.
    - gt: Tensor, target values.
    - beta: float, controls the point where the loss changes from L2 to L1.
    - alpha: float, base weight for the L2 region (small differences).
    - gamma: float, base weight for the L1 region (large differences).
    - epsilon: float, small constant to improve numerical stability.

    Returns:
    - loss: Tensor, adaptive smooth L1 loss.
    """
def l1_loss(network_output, gt, beta=0.168, alpha=0.52, gamma=1.0, epsilon=1e-6):

    diff = torch.abs(network_output - gt) + epsilon  # Add epsilon for stability

    # Dynamically adjust alpha and gamma based on the magnitude of the error
    dynamic_alpha = alpha / (1 + torch.log(1 + diff))  # Reduce alpha for large differences
    dynamic_gamma = gamma * (1 - torch.exp(-diff))    # Increase gamma for large differences

    # Calculate loss with weighted L2 and L1 regions
    loss = torch.where(
        diff < beta,
        dynamic_alpha * (diff ** 2) / (2 * beta),  # Adaptive L2 region
        dynamic_gamma * (diff - 0.5 * beta)       # Adaptive L1 region
    )
    
    return loss.mean()



def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

