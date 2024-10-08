import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def generate_data(f, x_range, t_range, x_steps, t_steps, reg_points, noise=0.0, vc_plot=False, device='cpu'):
    x_k = torch.linspace(x_range[0], x_range[1], x_steps, device=device) 
    t_k = torch.linspace(t_range[0], t_range[1], t_steps, device=device) 

    X_k, T_k = torch.meshgrid(x_k, t_k, indexing='xy')

    X_pde_flat = X_k.reshape(-1, 1)
    T_pde_flat = T_k.reshape(-1, 1)

    input_k = torch.cat([X_pde_flat, T_pde_flat], dim=1)

    x_r = torch.rand(reg_points, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    t_r = torch.rand(reg_points, 1, device=device) * (t_range[1] - t_range[0]) + t_range[0]
    
    input_r = torch.cat([x_r, t_r], dim=1)

    with torch.no_grad():
        U_r = f(input_r)

        if noise > 0.0:
            U_r_std = torch.std(U_r)  
            U_r = U_r + noise * U_r_std * torch.randn_like(U_r)

    input_k = input_k.requires_grad_(True)
    input_r = input_r.requires_grad_(True)
    U_r = U_r.requires_grad_(True)

    if vc_plot:
        return input_k, input_r, U_r, t_k
    else:
        return input_k, input_r, U_r


def plot_r(f1, model, input_k, t, x_range, t_range):
    input_k_copy = input_k.clone()
    t_copy = t.clone()
    
    U = f1(input_k_copy)
    U_pred, _, _, _ = model.forward(input_k_copy)

    absolute_error = torch.abs(U - U_pred)

    x_steps = input_k_copy.shape[0] // t_copy.shape[0]
    t_steps = t_copy.shape[0]

    U = U.reshape(x_steps, t_steps).detach().cpu().numpy()
    U_pred = U_pred.reshape(x_steps, t_steps).detach().cpu().numpy()
    absolute_error = absolute_error.reshape(x_steps, t_steps).detach().cpu().numpy()

    x = torch.linspace(x_range[0], x_range[1], x_steps).detach().cpu().numpy()
    t = torch.linspace(t_range[0], t_range[1], t_steps).detach().cpu().numpy()
    X, T = np.meshgrid(x, t)

    plt.figure(figsize=(21, 6))

    plt.subplot(1, 3, 1)
    plt.contourf(T, X, U, cmap='viridis') 
    plt.colorbar()
    plt.title(r'$U_{\mathrm{true}}$', fontsize=20)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$x$', fontsize=16)

    plt.subplot(1, 3, 2)
    plt.contourf(T, X, U_pred, cmap='viridis') 
    plt.colorbar()
    plt.title(r'$U_{\mathrm{pred}}$', fontsize=20)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$x$', fontsize=16)

    plt.subplot(1, 3, 3)
    plt.contourf(T, X, absolute_error, cmap='magma') 
    plt.colorbar()
    plt.title(r'$|\mathrm{Absolute\ Error}|$', fontsize=20)
    plt.xlabel(r'$t$', fontsize=16)
    plt.ylabel(r'$x$', fontsize=16)

    plt.tight_layout()
    plt.show()