import numpy as np
import matplotlib.pyplot as plt
import torch
import re
import os
import matplotlib.font_manager as fm
from matplotlib import font_manager

def compute_error(W, TestR, TestY, l0_penalty):
    steps_t = TestR.shape[0]
    err = 0
    for i in range(steps_t):
        err += np.linalg.norm(TestY[i, :, 0] - TestR[i, :, :].dot(W[:, i]), 2)
    
    err /= steps_t
    err += l0_penalty * np.count_nonzero(np.linalg.norm(W, axis=1))

    return err

def compute_error2(W, TestR, TestY, l0_penalty=None, epsilon=1e-5):
    steps_t = TestR.shape[0] 
    groups = W.shape[0] 
    points = TestR.shape[1]  
    
    rss = 0
    for i in range(steps_t):
        residual = TestY[i, :, 0] - TestR[i, :, :].dot(W[:, i])
        rss += np.linalg.norm(residual)**2
    
    k = np.count_nonzero(W) / steps_t
    
    N = points * steps_t
    
    error = N * np.log(rss / N + epsilon) + 2 * k + (2 * k**2 + 2 * k) / (N - k - 1)
    
    if l0_penalty is not None:
        error += l0_penalty * k
    
    return error


def GroupRidge(steps_t, lam, biginds, X, y, W, dtype):
    for i in range(steps_t):
        if lam != 0:
            W[biginds, i] = np.linalg.lstsq(
                X[i, :, :][:, biginds].T.dot(X[i, :, :][:, biginds]) + lam * np.eye(len(biginds), dtype=dtype),
                X[i, :, :][:, biginds].T.dot(y[i, :, 0]),
                rcond=None
            )[0]
        else:
            W[biginds, i] = np.linalg.lstsq(
                X[i, :, :][:, biginds],
                y[i, :, 0],
                rcond=None
            )[0]


def TrainGroupSTRidge(R, Ut, lam, d_tol, maxit=50, STR_iters=20, l0_penalty=None, normalize=2, split=0.8, print_result=False, type='real'):
    """
    This function trains a Group-STRidge predictor with support for real and complex types.
    """

    dtype = np.complex128 if type == 'complex' else np.float64

    np.random.seed(0) 
    steps_t, points, groups = R.shape

    train = np.random.choice(points, int(points * split), replace=False)
    test = [i for i in np.arange(points) if i not in train]

    TrainR = R[:, train, :]
    TestR = R[:, test, :]
    TrainY = Ut[:, train, :]
    TestY = Ut[:, test, :]

    d_tol = float(d_tol)
    epsilon = d_tol
    tol = d_tol
    if l0_penalty is None:
        l0_penalty = 0.001 * np.linalg.cond(R.reshape(steps_t * points, groups))

    W_best = np.zeros((groups, steps_t), dtype=dtype)
    W = np.zeros((groups, steps_t), dtype=dtype)

    for i in range(steps_t):
        W_best[:, i] = np.linalg.lstsq(TrainR[i, :, :], TrainY[i, :, 0], rcond=None)[0]

    err_best = compute_error(W_best, TestR, TestY, l0_penalty)
    tol_best = 0

    for iter in range(maxit + 1):
        W = GroupSTRidge(TrainR, TrainY, lam, STR_iters, tol, normalize, type=type)
        if W is None:
            break
        err = compute_error(W, TestR, TestY, l0_penalty)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            W_best = W.copy()
            tol_best = tol
            tol += d_tol
        else:
            tol = max([0, tol - 2 * d_tol])
            d_tol = 2 * d_tol / (maxit - iter + 1)
            tol += d_tol

    if print_result:
        print("Optimal tolerance:", tol_best)
        print("Best error:", err_best)
        print("iter:", iter)

    W_best[np.where(np.sum(np.abs(W_best), axis=1) < epsilon)[0], :] = 0
    biginds = np.where(np.sum(np.abs(W_best), axis=1) > epsilon)[0]
    GroupRidge(steps_t, 0, biginds, TrainR, TrainY, W_best, dtype)
    return W_best


def GroupSTRidge(X0, y, lam, maxit, tol, normalize=2, print_results=False, type='real'):
    dtype = np.complex128 if type == 'complex' else np.float64

    steps_t, points, groups = X0.shape
    X = np.zeros((steps_t, points, groups), dtype=dtype)
    Mreg = np.zeros((steps_t, groups), dtype=dtype)

    for i in range(steps_t):
        for j in range(groups):
            norm_factor = np.linalg.norm(X0[i, :, j], ord=normalize)
            if norm_factor != 0:
                Mreg[i, j] = 1.0 / norm_factor
                X[i, :, j] = Mreg[i, j] * X0[i, :, j]
            else:
                Mreg[i, j] = 1.0 
                X[i, :, j] = X0[i, :, j]

    W = np.zeros((groups, steps_t), dtype=dtype)
    for i in range(steps_t):
        if lam != 0:
            W[:, i] = np.linalg.lstsq(X[i, :, :].T.dot(X[i, :, :]) + lam * np.eye(groups, dtype=dtype),
                                      X[i, :, :].T.dot(y[i, :, 0]), rcond=None)[0]
        else:
            W[:, i] = np.linalg.lstsq(X[i, :, :], y[i, :, 0], rcond=None)[0]

    num_relevant = groups
    row_norms = np.linalg.norm(W, axis=1) 
    biginds = np.where(row_norms > tol)[0]

    # Threshold and continue
    for j in range(maxit):
        smallinds = np.where(row_norms < tol)[0]
        new_biginds = [i for i in range(groups) if i not in smallinds]

        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        if len(new_biginds) == 0:
            if j == 0:
                return W
            else:
                break
        biginds = new_biginds

        W[smallinds, :] = 0

        GroupRidege(steps_t, lam, biginds, X, y, W, dtype)

    if biginds != []:
        W[biginds, i] = np.linalg.lstsq(X[i, :, :][:, biginds], y[i, :, 0], rcond=None)[0]

    if normalize != 0:
        W = np.multiply(Mreg.T, W)

    return W



def plot_vc(w, d_tol, rhs_des, t, filename):
    """
    Plot the evolution of all coefficients over time and display the d_tol parameter in the title.
    Save the plot as a PDF in a 'vc_plot' folder.
    
    Parameters:
    - w: Coefficient matrix with shape (groups, steps_t)
    - d_tol: The d_tol parameter used in TrainSTRidge
    - rhs_des: A list used to generate labels
    - t: Array of time steps
    - filename: The name for the saved plot (without extension)
    """
    
    t = t.cpu().detach().numpy()
    groups, steps_t = w.shape
    
    labels = []
    for i in range(groups):
        if rhs_des[i] == '':
            labels.append('1')
        else:
            labels.append(rf'${rhs_des[i]}$')

    folder_path = 'vc_plot'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.figure(figsize=(15, 8))

    for i in range(groups):
        if not np.all(w[i, :] == 0):  
            plt.plot(t, w[i, :], label=labels[i])

    # Set labels and title
    plt.xlabel(r'$\mathbf{x}$', fontsize=22)  
    plt.ylabel(r'Coefficient Value', fontsize=22)  
    plt.title(r'Coefficient Evolution Over Time with $\mathbf{t}$' + '\n' + r'($\mathrm{d\_tol=%.2f}$)' % d_tol, fontsize=24)

    plt.legend(loc='best', fontsize=18) 
    
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16) 

    plt.grid(True)
    
    file_path = os.path.join(folder_path, f'{filename}.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    
    plt.show()

    print(f'Plot saved as {file_path}')


def compute_lib(input, U, rhs_des, type='real'):

    if type == 'real':
        U_x = torch.autograd.grad(U, input, grad_outputs=torch.ones_like(U), create_graph=True)[0][:,[0]]
        U_t = torch.autograd.grad(U, input, grad_outputs=torch.ones_like(U), create_graph=True)[0][:,[1]]
        U_xx = torch.autograd.grad(U_x, input, grad_outputs=torch.ones_like(U_x), create_graph=True)[0][:,[0]]
        U_xxx = torch.autograd.grad(U_xx, input, grad_outputs=torch.ones_like(U_xx), create_graph=True)[0][:,[0]]
        U_xxxx = torch.autograd.grad(U_xxx, input, grad_outputs=torch.ones_like(U_xxx), create_graph=True)[0][:,[0]]
        
        term_map_real = {
            'u': U, 
            'u_{x}': U_x,
            'u_{xx}': U_xx,
            'u_{xxx}': U_xxx,
            'u_{xxxx}': U_xxxx,
            'x': input[:, [0]],
            't': input[:, [1]],
        }

        lib_real = []
        
        for term in rhs_des:
            if term == '':
                lib_real.append(torch.ones_like(U))  
            else:
                components = re.findall(r'(?:u(?:_\{\w+\})?|x|t)\^?\d*', term)
                
                result_real = torch.ones_like(U)
                
                for component in components:
                    if '^' in component:
                        base, exponent = component.split('^')
                        exponent = int(exponent)
                        result_real *= term_map_real[base]**exponent
                    else:
                        result_real *= term_map_real[component]
                
                lib_real.append(result_real)
        
        return lib_real, U_t

    elif type == 'complex':
        U_real = U[:, [0]]
        U_imag = U[:, [1]]

        U_x_real = torch.autograd.grad(U_real, input, grad_outputs=torch.ones_like(U_real), create_graph=True)[0][:,[0]]
        U_t_real = torch.autograd.grad(U_real, input, grad_outputs=torch.ones_like(U_real), create_graph=True)[0][:,[1]]
        U_xx_real = torch.autograd.grad(U_x_real, input, grad_outputs=torch.ones_like(U_x_real), create_graph=True)[0][:,[0]]
        U_xxx_real = torch.autograd.grad(U_xx_real, input, grad_outputs=torch.ones_like(U_xx_real), create_graph=True)[0][:,[0]]
        U_xxxx_real = torch.autograd.grad(U_xxx_real, input, grad_outputs=torch.ones_like(U_xxx_real), create_graph=True)[0][:,[0]]

        U_x_imag = torch.autograd.grad(U_imag, input, grad_outputs=torch.ones_like(U_imag), create_graph=True)[0][:,[0]]
        U_t_imag = torch.autograd.grad(U_imag, input, grad_outputs=torch.ones_like(U_imag), create_graph=True)[0][:,[1]]
        U_xx_imag = torch.autograd.grad(U_x_imag, input, grad_outputs=torch.ones_like(U_x_imag), create_graph=True)[0][:,[0]]
        U_xxx_imag = torch.autograd.grad(U_xx_imag, input, grad_outputs=torch.ones_like(U_xx_imag), create_graph=True)[0][:,[0]]
        U_xxxx_imag = torch.autograd.grad(U_xxx_imag, input, grad_outputs=torch.ones_like(U_xxx_imag), create_graph=True)[0][:,[0]]

        # Include 'x' and 't' in the term maps
        term_map_real = {
            'u': U_real,
            'u_{x}': U_x_real,
            'u_{xx}': U_xx_real,
            'u_{xxx}': U_xxx_real,
            'u_{xxxx}': U_xxxx_real,
            '|u|': torch.sqrt(U_real**2 + U_imag**2),
            '|u|^2': (U_real**2 + U_imag**2),
            '|u|^3': (U_real**2 + U_imag**2) ** 1.5,
            'x': input[:, [0]],
            't': input[:, [1]],
        }

        term_map_imag = {
            'u': U_imag,
            'u_{x}': U_x_imag,
            'u_{xx}': U_xx_imag,
            'u_{xxx}': U_xxx_imag,
            'u_{xxxx}': U_xxxx_imag,
            '|u|': torch.sqrt(U_real**2 + U_imag**2),
            '|u|^2': (U_real**2 + U_imag**2),
            '|u|^3': (U_real**2 + U_imag**2) ** 1.5,
            'x': torch.zeros_like(U_imag),  
            't': torch.zeros_like(U_imag), 
        }

        lib = []
        
        for term in rhs_des:
            is_imaginary = 'i' in term 
            term = term.replace('i', '') 
            
            if term == '':
                lib.append(torch.ones_like(U_real) + 1j * torch.ones_like(U_imag)) 
            else:
                # Parse the term into components
                components = re.findall(r'(?:\|?u(?:_\{\w+\})?\|?|x|t)\^?\d*', term)
                
                result_real = torch.ones_like(U_real)
                result_imag = torch.ones_like(U_imag)
                
                for component in components:
                    if component.startswith('|u|'):
                        # Handle absolute value cases
                        if '^' in component:
                            exponent = int(component.split('^')[-1])
                            abs_term = torch.sqrt(U_real**2 + U_imag**2) ** exponent
                        else:
                            abs_term = torch.sqrt(U_real**2 + U_imag**2)
                        result_real *= abs_term
                        result_imag *= abs_term
                    else:
                        # Handle standard u, u_x, u_xx, x, t, etc.
                        if '^' in component:
                            base, exponent = component.split('^')
                            exponent = int(exponent)
                            result_real *= term_map_real[base]**exponent
                            result_imag *= term_map_imag[base]**exponent
                        else:
                            result_real *= term_map_real[component]
                            result_imag *= term_map_imag[component]
                
                # If the term includes imaginary unit 'i', multiply the result by 1j
                if is_imaginary:
                    lib.append((result_real + 1j * result_imag) * 1j)
                else:
                    lib.append(result_real + 1j * result_imag)
        
        U_t = (U_t_real + 1j * U_t_imag)

        U_t = 1j * U_t
        
        return lib, U_t


def kan2Ridge(lib, U_t, x_steps, t_steps):
    lib_concat = torch.cat(lib, dim=1)
    R = torch.stack([lib_concat[i*x_steps:(i+1)*x_steps,:] for i in range(t_steps)], dim=0)
    Ut = torch.stack([U_t[i*x_steps:(i+1)*x_steps,:] for i in range(t_steps)], dim=0)
    R = R.cpu().detach().numpy()
    Ut = Ut.cpu().detach().numpy()
    return R, Ut
    

