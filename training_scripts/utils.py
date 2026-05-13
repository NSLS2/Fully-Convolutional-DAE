#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import torch.nn as nn
from Nets_FC import AutoEncoder_2D


def symmetry_enf(Y_out, Y_t):
    """
    MSE loss that also enforces symmetry for permutations of [t1,t2]. Not used in the final models.
    """
    return torch.mean((Y_out - Y_t) ** 2) + torch.mean(
        (Y_out - torch.transpose(Y_out, 2, 3)) ** 2
    )


def elastic_loss(X_in, Y_out, Y_t, alpha=0.5):
    """
    Weighted MSE loss between model output and both target & input.
    """
    return alpha * torch.mean((Y_out - Y_t) ** 2) + (1 - alpha) * torch.mean(
        (Y_out - X_in) ** 2
    )


def get_loss_function(loss_name):
    """
    Returns a loss function that always takes (X_in, Y_out, Y_target).
    """
    if loss_name == "mse":
        base_loss = nn.MSELoss()
        return lambda X, Y_out, Y_target: base_loss(Y_out, Y_target)
    elif loss_name == "mae":
        base_loss = nn.L1Loss()
        return lambda X, Y_out, Y_target: base_loss(Y_out, Y_target)
    elif loss_name == "smoothl1":
        base_loss = nn.SmoothL1Loss()
        return lambda X, Y_out, Y_target: base_loss(Y_out, Y_target)
    elif loss_name == "elastic":
        return elastic_loss
    elif loss_name == "symmetry":
        return symmetry_enf
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def setup_nn(ksize=3, lr=0.001, savefile=None, device="cpu", loss_name="mse"):
    """
    Initializes the model, loss function, optimizer and scheduler for training.
    """

    if savefile is not None:
        net = torch.load(savefile)
        net.eval()

    else:
        print(f"Running on {device}")
        ksizes = [ksize] * 5
        channels = [1, 1, 4, 8, 16, 32]
        strides_list = [1, 1, 1, 1, 1]
        net = AutoEncoder_2D(
            channels_list=channels, ksize=ksizes, strides_list=strides_list
        ).to(device)

    loss_function = get_loss_function(loss_name)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9995)

    return net, optimizer, scheduler, loss_function


def set_seed(seed):
    "Sets up all random seeds to ensure reproducibility."

    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
