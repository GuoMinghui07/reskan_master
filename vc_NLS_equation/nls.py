import numpy as np
import sympy as sp
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

def set_seed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def compute_nls(params, mode='abs'):
    """
    Compute the one-soliton solution for the nonlinear Schrödinger equation.
    
    Parameters:
        params (dict): Dictionary containing the parameters for the equation.
        mode (str): The mode of output. 'abs' returns |u|^2, 'complex' returns u as a complex number.
        
    Returns:
        function: A function that computes the desired output given t and x.
    """
    # Extract parameters from the dictionary
    b_func = params["b_func"]
    l_func = params["l_func"]
    d_func = params["d_func"]
    eta = params["eta"]
    eta0 = params["eta0"]
    C0 = params["C0"]
    C1 = params["C1"]

    t, x = sp.symbols('t x', real=True)
    
    b_x = b_func(x)
    l_x = l_func(x)
    d_x = d_func(x)
    
    theta = eta * t + sp.I * eta**2 * sp.integrate(b_x, x) + eta0
    theta_conj = sp.conjugate(theta)
    
    # One-soliton solution u(x, t)
    u = C1 * sp.exp(-sp.integrate(d_x, x)) * sp.exp(theta) / (1 + (1 / (2 * (eta + sp.conjugate(eta))**2)) * sp.exp(theta + theta_conj))
    
    if mode == 'complex':
        # Return the complex function u(t, x)
        return sp.lambdify((t, x), u, "numpy")
    elif mode == 'abs':
        # Compute |u|^2
        intensity = sp.simplify(sp.Abs(u)**2)
        return sp.lambdify((t, x), intensity, "numpy")
    else:
        raise ValueError("Invalid mode. Choose either 'abs' or 'complex'.")




def generate_data(t_range, x_range, t_steps, x_steps, point_r, params, device='cpu', vc_plot=True, noise=0., fun_type = 'one'):
    """
    Generates data for the Schrödinger equation.
    
    Parameters:
        t_range (tuple): The range of t (start, end).
        x_range (tuple): The range of x (start, end).
        t_steps (int): The number of steps in t.
        x_steps (int): The number of steps in x.
        point_r (int): The number of random points to sample.
        params (dict): Parameters for the nonlinear Schrödinger equation.
        device (str): The device to use for the tensors ('cpu' or 'cuda').
    
    Returns:
        input_r (Tensor): Randomly sampled points within the range (t, x).
        input_k (Tensor): Regular grid of points (t, x).
        U_r (Tensor): The complex values of the soliton equation evaluated at input_r, with gradient tracking.
    """
    if fun_type == 'one':
        sol = compute_nls(params, mode='complex')
    elif fun_type == 'two':
        sol = compute_nls_two_soliton(params, mode='complex')
    
    t_r = np.random.uniform(t_range[0], t_range[1], point_r)
    x_r = np.random.uniform(x_range[0], x_range[1], point_r)
    input_r_np = np.stack((t_r, x_r), axis=-1)
    
    t_k = np.linspace(t_range[0], t_range[1], t_steps)
    x_k = np.linspace(x_range[0], x_range[1], x_steps)
    T, X = np.meshgrid(t_k, x_k)
    input_k_np = np.stack((T.flatten(), X.flatten()), axis=-1)
    
    U_r_np = sol(input_r_np[:, 0], input_r_np[:, 1])
    if noise > 0.0:
        U_r_std = np.std(U_r_np)
        U_r_np = U_r_np + noise * U_r_std * np.random.randn(*U_r_np.shape)
    
    input_r = torch.tensor(input_r_np, dtype=torch.float32, requires_grad=True, device=device)
    input_k = torch.tensor(input_k_np, dtype=torch.float32, requires_grad=True, device=device)
    
    U_r_real = torch.tensor(np.real(U_r_np), dtype=torch.float32, requires_grad=True, device=device).unsqueeze(-1)
    U_r_imag = torch.tensor(np.imag(U_r_np), dtype=torch.float32, requires_grad=True, device=device).unsqueeze(-1)
    U_r = torch.complex(U_r_real, U_r_imag)

    if vc_plot == True:
        return input_r, input_k, U_r, torch.tensor(x_k, dtype=torch.float32, requires_grad=True, device=device)
    else:
        return input_r, input_k, U_r


def plot_solutions(t_range, x_range, params1, model, fun_type='one', save_fig=False, file_name='plot.pdf'):

    font_path = 'Times New Roman.ttf' 
    font_prop = fm.FontProperties(fname=font_path)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font_prop.get_name()],
        'axes.titlesize': 25,
        'axes.labelsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'font.size': 10,
        'xtick.labelbottom': True,
        'ytick.labelleft': True,
        'text.usetex': False,
        'mathtext.rm': font_prop.get_name(),
        'mathtext.it': font_prop.get_name(),
        'mathtext.bf': font_prop.get_name(),
        'mathtext.default': 'rm',
        'axes.unicode_minus': False,
    })

    t_values = np.linspace(t_range[0], t_range[1], 100)
    x_values = np.linspace(x_range[0], x_range[1], 100)
    T, X = np.meshgrid(t_values, x_values)
    
    T_tensor = torch.tensor(T, dtype=torch.float32, device=model.device).reshape(-1, 1).requires_grad_(True)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=model.device).reshape(-1, 1).requires_grad_(True)
    input_tensor = torch.cat([T_tensor, X_tensor], dim=1)

    if fun_type == 'one':
        intensity_func = compute_nls(params1)
    elif fun_type == 'two':
        intensity_func = compute_nls_two_soliton(params1)

    intensity_values = intensity_func(T, X)

    prediction, _, _, _ = model.forward(input_tensor, type='complex')

    prediction_magnitude = (prediction[:, [0]]**2 + prediction[:, [1]]**2).cpu().detach().numpy().reshape(T.shape)

    absolute_error = np.abs(prediction_magnitude - intensity_values)

    fig = plt.figure(figsize=(20, 8))

    def set_tick_formatting(ax, tick_size=22, pad=2, max_locator=2):
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_locator))
        ax.yaxis.set_major_locator(plt.MaxNLocator(max_locator))
        ax.zaxis.set_major_locator(plt.MaxNLocator(max_locator))
        ax.tick_params(axis='x', pad=pad, labelsize=tick_size)
        ax.tick_params(axis='y', pad=pad, labelsize=tick_size)
        ax.tick_params(axis='z', pad=pad, labelsize=tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(font_prop)
            label.set_fontsize(tick_size)

    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(T, X, intensity_values, cmap='coolwarm', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    cbar1.ax.yaxis.set_tick_params(labelsize=25)
    cbar1.ax.set_yticklabels(cbar1.ax.get_yticks(), fontproperties=font_prop)  # 设置颜色条字体
    ax1.set_title('Real Solution', fontproperties=font_prop)
    ax1.set_xlabel('t', fontproperties=font_prop)
    ax1.set_ylabel('x', fontproperties=font_prop)
    set_tick_formatting(ax1, tick_size=22)
    ax1.view_init(elev=30, azim=120)

    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(T, X, prediction_magnitude, cmap='coolwarm', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    cbar2.ax.yaxis.set_tick_params(labelsize=25)
    cbar2.ax.set_yticklabels(cbar2.ax.get_yticks(), fontproperties=font_prop)  # 设置颜色条字体
    ax2.set_title('Predicted Solution', fontproperties=font_prop)
    ax2.set_xlabel('t', fontproperties=font_prop)
    ax2.set_ylabel('x', fontproperties=font_prop)
    set_tick_formatting(ax2, tick_size=22)
    ax2.view_init(elev=30, azim=120)

    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(T, X, absolute_error, cmap='coolwarm', edgecolor='none', alpha=0.9, rstride=1, cstride=1)
    cbar3 = fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
    cbar3.ax.yaxis.set_tick_params(labelsize=25)
    cbar3.ax.set_yticklabels(cbar3.ax.get_yticks(), fontproperties=font_prop)  # 设置颜色条字体
    ax3.set_title('Absolute Error', fontproperties=font_prop)
    ax3.set_xlabel('t', fontproperties=font_prop)
    ax3.set_ylabel('x', fontproperties=font_prop)
    set_tick_formatting(ax3, tick_size=22)
    ax3.view_init(elev=30, azim=120)

    if save_fig:
        plt.savefig(file_name, format='pdf')

    plt.tight_layout()
    plt.show()

    whole_params_set = [
    {"b_func": lambda x: 1, "l_func": lambda x: 1, "d_func": lambda x: 0, "eta": 1 - 0.2*sp.I, "eta0": 1 - 2*sp.I, "C0": 1, "C1": 1},
    {"b_func": lambda x: 1, "l_func": lambda x: 1, "d_func": lambda x: 0, "eta": 1.4 - 0.1*sp.I, "eta0": 3 - 3*sp.I, "C0": 1, "C1": 1},
    {"b_func": lambda x: 1, "l_func": lambda x: sp.exp(-0.4*sp.cos(x)), "d_func": lambda x: 0.2*sp.sin(x), "eta": 1.8, "eta0": 0.05, "C0": 1, "C1": 1},
    {"b_func": lambda x: 1, "l_func": lambda x: sp.exp(0.4*sp.sin(x)), "d_func": lambda x: 0.2*sp.cos(x), "eta": 1.8, "eta0": 0.05, "C0": 1, "C1": 1},
    {"b_func": lambda x: 0.5*sp.sin(0.5*x), "l_func": lambda x: 0.5*sp.sin(0.5*x), "d_func": lambda x: 0, "eta": 1.8 + 0.6*sp.I, "eta0": 0, "C0": 1, "C1": 1},
    {"b_func": lambda x: 0.5*sp.cos(0.5*x), "l_func": lambda x: 0.5*sp.cos(0.5*x), "d_func": lambda x: 0, "eta": 1.8 + 0.6*sp.I, "eta0": 0, "C0": 1, "C1": 1}
]


    