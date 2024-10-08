import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import random
from kan import *
from .Group_STR import *
from tqdm import tqdm
from kan.compiler import kanpiler
from sympy import *
import gc

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
seed = 6666
set_seed(seed)

class Block(nn.Module):
    def __init__(self, Nd, Nh):
        super(Block, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(Nh):
            self.layers.append(nn.Linear(Nd, Nd))
            self.layers.append(nn.Tanh())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ResNet(nn.Module):
    def __init__(self, Nd, Nh, Nb, input_dim, output_dim):
        super(ResNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, Nd)
        self.blocks = nn.ModuleList([Block(Nd, Nh) for _ in range(Nb)]) 
        self.linear_out = nn.Linear(Nd, output_dim)  
        self.initialize_weights()

    def forward(self, x):
        x = self.linear_in(x)
        for block in self.blocks:
            identity = x
            x = block(x)
            x = x + identity 
        x = self.linear_out(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


class VC_PIKAN(nn.Module):
    def __init__(self, Nd, Nh, Nb, rhs_des, range_x, range_t, grid=5, k=3, input_dim=2, output_dim=1, seed=1234, mode='resnet', device='cpu', scale_base_mu=0., base_fun='identity'):
        super(VC_PIKAN, self).__init__()
        self.Nd, self.Nh, self.Nb = Nd, Nh, Nb
        self.range_x = range_x
        self.range_t = range_t
        self.rhs_des = rhs_des
        self.rhs_des_label = [f'${term}$' if term else '$1$' for term in self.rhs_des]
        self.row = len(rhs_des)
        self.mode = mode
        self.biginds = np.arange(0, len(rhs_des))
        self.resnet = ResNet(Nd, Nh, Nb, input_dim, output_dim).to(device) 
        self.main_kan = MultKAN(width=[input_dim,20,20,output_dim], grid=5, k=3, grid_range=[-1,1], seed=seed).to(device).speed()
        self.kan = MultKAN(width=[1, len(rhs_des)], grid=grid, k=k, base_fun=base_fun, grid_range=[range_t[0], range_t[1]], seed=seed, scale_base_mu=scale_base_mu).to(device) 
        self.device = device 
        self.lb = torch.tensor([range_x[0], range_t[0]], device=device)
        self.ub = torch.tensor([range_x[1], range_t[1]], device=device)

    def normalize(self, input):
        return 2.0 * (input - self.lb) / (self.ub - self.lb) - 1.0

    def forward(self, input, type='real'):
        input_normalized = self.normalize(input)

        if self.mode == 'resnet':
            U = self.resnet(input_normalized)
        elif self.mode == 'kan':
            U = self.main_kan(input_normalized)
            
        if type == 'real':
            lib, U_t = compute_lib(input, U, self.rhs_des, type)
            output = self.kan(input[:, [1]])
            return U, lib, U_t, output
        elif type == 'complex':
            lib, U_t = compute_lib(input, U, self.rhs_des, type)
            lib_real = [term.real for term in lib]
            lib_imag = [term.imag for term in lib]
            U_t_real = U_t.real
            U_t_imag = U_t.imag
            output = self.kan(input[:, [1]])  
            return U, (lib_real, lib_imag), (U_t_real, U_t_imag), output

    
    def Loss(self, input_r, input_k, label, lam, lamb_l1, lamb_coef, lamb_coefdiff, lamb_entropy, alpha, normal, reg_metric, type='real'):
        if type == 'real':
            U_r, _, _, _ = self.forward(input_r, type)
            _, lib_k, Ut_k, output_k = self.forward(input_k, type)
        elif type == 'complex':
            U_r, _, _, _ = self.forward(input_r, type)
            _, (lib_real_k, lib_imag_k), (Ut_real_k, Ut_imag_k), output_k = self.forward(input_k, type)

        if type == 'real':
            residual_pde = 0
        elif type == 'complex':
            residual_pde_r = 0
            residual_pde_i = 0
            
        normalization_factors = []

        if type == 'real':
            for i in range(len(lib_k)):
                if normal:
                    lib_norm = torch.norm(lib_k[i], p=2) 
                    normalization_factors.append(lib_norm.item())  
                    lib_normalized = lib_k[i] / lib_norm 
                else:
                    lib_normalized = lib_k[i]
                    normalization_factors.append(1.0) 

                residual_pde += output_k[:, [i]] * lib_normalized 

            residual_pde += Ut_k

        elif type == 'complex':
            for i in range(len(lib_real_k)):
                if normal:
                    lib_norm_r = torch.norm(lib_real_k[i], p=2)
                    lib_norm_i = torch.norm(lib_imag_k[i], p=2)
                    normalization_factors.append((lib_norm_r.item(), lib_norm_i.item()))

                    lib_real_normalized = lib_real_k[i] / lib_norm_r
                    lib_imag_normalized = lib_imag_k[i] / lib_norm_i
                else:
                    lib_real_normalized = lib_real_k[i]
                    lib_imag_normalized = lib_imag_k[i]
                    normalization_factors.append((1.0, 1.0))

                residual_pde_r += output_k[:, [i]] * lib_real_normalized
                residual_pde_i += output_k[:, [i]] * lib_imag_normalized

            residual_pde_r += Ut_real_k
            residual_pde_i += Ut_imag_k

        if type == 'real':
            loss_pde = torch.mean(residual_pde ** 2) 
        elif type == 'complex':
            loss_pde = torch.mean(residual_pde_r ** 2 + residual_pde_i ** 2) 

        loss_reg = self.kan.get_reg(
            reg_metric=reg_metric,
            lamb_l1=lamb_l1,
            lamb_entropy=lamb_entropy,
            lamb_coef=lamb_coef,
            lamb_coefdiff=lamb_coefdiff
        )

        loss_kan = loss_pde + lam * loss_reg

        if type == 'real':
            loss_data = torch.mean((U_r - label) ** 2)
        elif type == 'complex':
            U_r_real = U_r[:, [0]]
            U_r_imag = U_r[:, [1]]
            label_real = label.real
            label_imag = label.imag

            real_error = (U_r_real - label_real) ** 2
            imag_error = (U_r_imag - label_imag) ** 2
            total_error = real_error + imag_error

            loss_data = torch.mean(total_error)


        total_loss = torch.log(loss_data + alpha * loss_kan)

        return total_loss, loss_data, loss_kan, loss_pde, loss_reg



    def fit(self, input_r, input_k, label, steps=100, lr=1, lam=0.001, lamb_l1=1., lamb_coef=0, lamb_coefdiff=0, lamb_entropy=2., alpha=1., normal=False, mode='all', reg_metric='edge_forward_spline_n', type='real', img_folder='./video', save_fig=False, save_fig_freq=1, save_every10step=False, save_loss=False, loss_filename='losses.txt'):
        if mode == 'all':
            self.para = self.parameters()
        elif mode == 'resnet':
            self.para = list(self.resnet.parameters())
        elif mode == 'kan':
            self.para = list(self.kan.parameters())
        optimizer = torch.optim.LBFGS(self.para, lr=lr, history_size=10, line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-32, tolerance_change=1e-32)
        self.input = input_k

        progress_bar = tqdm(range(steps))
        total_loss, loss_data, loss_kan, loss_pde, loss_reg = None, None, None, None, None

        def closure():
            nonlocal total_loss, loss_data, loss_kan, loss_pde, loss_reg
            optimizer.zero_grad()
            total_loss, loss_data, loss_kan, loss_pde, loss_reg = self.Loss(input_r, input_k, label, lam, lamb_l1, lamb_coef, lamb_coefdiff, lamb_entropy, alpha, normal, reg_metric, type)
            total_loss.backward(retain_graph=True)
            return total_loss

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        if save_loss:
            loss_file = open(loss_filename, 'a')
            loss_file.write(f"{img_folder}\n")

        print(f"Training on {self.device}")

        try:
            for _ in progress_bar:
                optimizer.step(closure)
                loss = closure().item()
                if save_fig and _ % save_fig_freq == 0:
                    self.kan.plot(folder=img_folder, scale=2, in_vars=['x'], out_vars=self.rhs_des_label,
                                varscale=3.5 / len(self.rhs_des), title="Step {}".format(_), beta=10)
                    plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                    plt.clf()
                    plt.close()
                    gc.collect()

                    if save_every10step and _ % 10 == 0:
                        name = img_folder + '/' + str(_)
                        os.makedirs(name)
                        self.kan.plot(folder=name, scale=2, in_vars=['x'], out_vars=self.rhs_des_label,
                                    varscale=3.5 / len(self.rhs_des), title="Step {}".format(_), beta=10)

                with torch.no_grad():
                    progress_bar.set_postfix({'loss': loss, 'loss_data': loss_data.item(), 'loss_pde': loss_pde.item(), 'loss_reg': loss_reg.item()})

                if save_loss:
                    loss_file.write(f"total_loss: {total_loss.item()}, loss_data: {loss_data.item()}, "
                    f"loss_kan: {loss_kan.item()}, loss_pde: {loss_pde.item()}, "
                    f"loss_reg: {loss_reg.item()}\n")

        finally:
            if save_loss:
                loss_file.close()


    def kan_plot(self, beta=10):
        self.kan.plot(scale=2, in_vars=['x'], out_vars=self.rhs_des_label, varscale= 3.5 / len(self.rhs_des), beta=beta)

    def G_STRidge(self, input_k, lam, d_tol, x_steps, t_steps, maxit=25, STR_iters=10, l0_penalty=None, normalize=2, split=0.8, print_result=False,type='real'):
        U,_,_,_ = self.forward(input_k, type=type)
        Mask = self.kan.act_fun[0].mask[0]
        self.kan.log_history('remove_edge')
        self.rhs_des_prune = [self.rhs_des[i] for i in range(len(Mask)) if Mask[i] == 1]
        lib, U_t = compute_lib(input_k, U, self.rhs_des_prune, type=type)
        R, Ut = kan2Ridge(lib, U_t, x_steps, t_steps)
        W = TrainGroupSTRidge(R, Ut, lam, d_tol, maxit=maxit, STR_iters=STR_iters, l0_penalty=l0_penalty, normalize=normalize, split=split, print_result=print_result, type=type)
        self.W = W
        return W

    def prune_kan(self, input_k, threshold=3e-2):
        self.forward(input_k)
        print("Original:")
        self.kan_plot()
        print("After pruning:")
        self.kan.prune_edge(threshold=threshold)
        self.kan_plot()

    def active_remove_edge(self,i):
        self.kan.remove_edge(0,0,i)
    
    def ridge_plot(self, d_tol, t, filename):
        plot_vc(self.W, d_tol, self.rhs_des_prune, t, filename)

    def plot_kan_components(self, t_values=None, save_fig=False, filename='kan_components.pdf'):
        """
        Plots all active components of self.kan with respect to the independent variable in one figure.

        Args:
            t_values (array-like, optional): Array of time values to evaluate. If None, generates 100 evenly spaced points over self.range_t.
            save_fig (bool, optional): Whether to save the figure. Defaults to False.
            filename (str, optional): Filename to save the figure if save_fig is True. Defaults to 'kan_components.pdf'.
        """
        if t_values is None:
            t_values = np.linspace(self.range_t[0], self.range_t[1], 100)
        t_tensor = torch.tensor(t_values, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.kan(t_tensor)
        outputs_np = outputs.cpu().numpy()

        # Get the mask from the activation function to identify active components
        Mask = self.kan.act_fun[0].mask[0].cpu().numpy()
        active_indices = np.where(Mask == 1)[0]

        # Get labels for active components
        active_labels = [self.rhs_des_label[i] for i in active_indices]

        plt.figure(figsize=(10, 6))
        for idx, label in zip(active_indices, active_labels):
            plt.plot(t_values, outputs_np[:, idx], label=label)
        plt.xlabel('t')
        plt.ylabel('KAN Outputs')
        plt.title('Active KAN Components vs Time')
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=200)
            plt.close()
        else:
            plt.show()


    




