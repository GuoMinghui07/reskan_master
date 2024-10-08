import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter

def plot_variable_coefficients(model, true_sol_dict, x_range, t_range, symbolic_model=None, save_fig=False, file_name='plot.pdf'):

    Mask = model.kan.act_fun[0].mask[0].cpu().numpy()
    active_indices = np.where(Mask == 1)[0]
    print("Active Indices:", active_indices)

    active_labels = [model.rhs_des_label[i] for i in active_indices]

    t_values = np.linspace(t_range[0], t_range[1], num=200)
    t_tensor = torch.tensor(t_values, dtype=torch.float32).unsqueeze(1).to(model.device)

    with torch.no_grad():
        outputs_model = model.kan(t_tensor)
    outputs_model_np = outputs_model.cpu().numpy()

    symbolic_formula_outputs = None
    if symbolic_model is not None:
        symbolic_formula_outputs = symbolic_model.kan.symbolic_formula()[0]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    font_path = os.path.join(current_dir, 'Times New Roman.ttf')
    font_prop = fm.FontProperties(fname=font_path)

    plt.rcParams.update({
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'font.size': 24,
        'axes.titlesize': 22,
        'axes.labelsize': 20
    })

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['mathtext.default'] = 'regular'

    plt.figure(figsize=(12, 8), dpi=120)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_colors = len(color_cycle)

    for idx_num, idx in enumerate(active_indices):
        label = active_labels[idx_num]
        color = color_cycle[idx_num % num_colors]

        model_prediction = outputs_model_np[:, idx]

        plt.plot(t_values, model_prediction, linestyle='--', color=color, linewidth=2, label=f'{label} - Before symbolic')

        if symbolic_formula_outputs is not None:
            symbolic_prediction = np.array([symbolic_formula_outputs[idx].subs('x_1', t).evalf() for t in t_values])
            plt.plot(t_values, symbolic_prediction, linestyle=':', color=color, linewidth=2, label=f'{label} - Symbolic')

        if idx in true_sol_dict:
            true_solution = true_sol_dict[idx](t_values)
            plt.plot(t_values, true_solution, linestyle='-', color=color, linewidth=2, label=f'{label} - True')
        else:
            print(f"No true solution available for index {idx}")

    plt.xlabel(r'$t$', fontsize=20, fontproperties=font_prop)
    plt.ylabel('Coefficient Value', fontsize=25, fontproperties=font_prop)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(24)

    plt.legend(fontsize=16, shadow=True, prop=font_prop)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_name, format='pdf')

    plt.show()