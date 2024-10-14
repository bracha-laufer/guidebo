import os
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
sns.set(style="whitegrid",font_scale=1.45)

def plot_init_configs(folder, name, scores_names, lambdas, scores):
    plt.figure()
    df_init = pd.DataFrame({scores_names[0]: scores[:,0],
                            scores_names[1]: scores[:,1],
                            r'$\lambda$': lambdas[:,0]})
    sns.scatterplot(df_init, x=scores_names[0], y=scores_names[1], hue=r'$\lambda$', s=75)
    plt.savefig(os.path.join(folder,f'init_configs_{name}.png')) 

def plot_BO(folder, name, scores_names, g, N0, alpha=None, ref_point=None, lower_risks=None, upper_risks=None):
    
    plt.figure()

    plt.scatter(g[:N0,0], g[:N0,1], c='b', s=45)
    plt.scatter(g[N0:,0], g[N0:,1], c='r', s=45)
    
    
    if name == 'Proposed':
        plt.scatter(ref_point[0], ref_point[1], c='#008000', marker='s', s=60)  
        y_min, y_max = np.min(g[:,1]), np.max(g[:,1])
        x_min, x_max = np.min(g[:,0]), np.max(g[:,0])
        plt.vlines(x=[lower_risks[-1,0], upper_risks[-1,0]], ymin=y_min, ymax=y_max, color='0.6', linestyles='dashed')
        if lower_risks.shape[1] == 2:
            plt.hlines(y=[lower_risks[-1,1], upper_risks[-1,1]], xmin=x_min, xmax=x_max, color='0.6', linestyles='dashed')
        else:
            plt.hlines(y=ref_point[1], xmin=x_min, xmax=x_max, color='0.6', linestyles='dashed')
    
    elif name == r'$\alpha$-Limit':
        y_min, y_max = np.min(g[:,1]), np.max(g[:,1])
        plt.vlines(x=[upper_risks[0]], ymin=y_min, ymax=y_max, color='0.6', linestyles='dashed')

    
    
    plt.ylabel(scores_names[1])
    plt.xlabel(scores_names[0])
    if alpha is None:
        file_name = f'selected_{name}.png'
    else:    
        file_name = f'selected_{name}_alpha_{alpha}.png'
    plt.savefig(os.path.join(folder,file_name))


def plot_res(folder, task, scores_names, res_df, init_name=''):

    n_scores = len(scores_names) 

    alphas1 = res_df[f'$\\alpha_{1}$'].unique()
    methods = res_df['Method'].unique()


#     for i in range(n_scores-1):
#         plt.figure()
#         sns.set(style="whitegrid",font_scale=1.5)
#         if i == 0:
#             alpha_i = alphas1
#         else:
#             alpha_i = len(alphas1)*list(res_df[f'$\\alpha_{i+1}$'].unique())
#         print(alpha_i)
#         plt.plot(alphas1, alpha_i, linestyle='--', color='black', label='diagonal')
#         sns.lineplot(x=f'$\\alpha_{1}$', y=scores_names[i], hue="Method", style="Method", markers=True, dashes=False, data=res_df, errorbar='sd')
#         plt.savefig(os.path.join(folder, f'{init_name}{task}_{scores_names[i]}_line.png'))  

#     plt.figure()
#     sns.lineplot(x=f'$\\alpha_{1}$', y=scores_names[-1], hue="Method", style="Method", markers=True,  dashes=False, data=res_df, errorbar='sd')
#     plt.savefig(os.path.join(folder, f'{init_name}{task}_{scores_names[-1]}_line.png'))                    
#     print(res_df.groupby([f'$\\alpha_{1}$', 'Method']).mean())    

#     res_df['Violations'] = (res_df[scores_names[0]]>res_df[f'$\\alpha_{1}$']).astype(int)

#     plt.figure()
#     sns.barplot(x=f'$\\alpha_{1}$', y='Violations', hue="Method", data=res_df, errorbar=None)
#     plt.savefig(os.path.join(folder, f'{init_name}{task}_violations.png'))    

#     for i in range(n_scores-1):
#         plt.figure(figsize=(7,6))
#         ax = sns.barplot(x=f'$\\alpha_{1}$', y=scores_names[i], hue="Method", data=res_df, errorbar="sd", err_kws={'linewidth': 1.5})

#         tot_len = len(alphas1)*len(methods) 
#         dd = 0.3
#         for j, a in enumerate(alphas1):
#             ax.axhline(y=a, color='r', linestyle='--', lw=2.5, xmin=(j)/len(alphas1)+0.5/tot_len, xmax=(j+1)/len(alphas1)-0.1/len(alphas1))

#         #plt.locator_params(nbins=len(alphas_list))
#         plt.savefig(os.path.join(folder, f'{init_name}{task}_{scores_names[i]}_bar.png'))  

    plt.figure(figsize=(7,6))
    default_palette = sns.color_palette()
    new_palette = default_palette #c*3 + [default_palette[1]]
    #new_palette = [default_palette[0]]*3 + [default_palette[1]]
    #new_palette = [default_palette[0], default_palette[6]]
    ax = sns.barplot(x=f'$\\alpha_{1}$', y=scores_names[-1], hue="Method", data=res_df, errorbar="sd", err_kws={'linewidth': 1.5}, palette=new_palette)
    
    # hatches = ["//", "x", ".", ""]
    # for bars, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles):
    #     handle.set_hatch(hatch)
    #     for bar in bars:
    #         bar.set_hatch(hatch)

    #plt.locator_params(nbins=len(alphas_list))
    plt.savefig(os.path.join(folder, f'{init_name}{task}_{scores_names[-1]}_bar.png')) 


def plot_res_per_budget(folder, task, scores_names, res_df, init_name=''):

    n_scores = len(scores_names) 

    alphas1 = res_df[f'$\\alpha_{1}$'].unique()
    methods = res_df['Method'].unique()


#     for i in range(n_scores-1):
#         plt.figure()
#         sns.set(style="whitegrid",font_scale=1.5)
#         if i == 0:
#             alpha_i = alphas1
#         else:
#             alpha_i = len(alphas1)*list(res_df[f'$\\alpha_{i+1}$'].unique())
#         print(alpha_i)
#         plt.plot(alphas1, alpha_i, linestyle='--', color='black', label='diagonal')
#         sns.lineplot(x='Budget', y=scores_names[i], hue="Method", style="Method", markers=True, dashes=False, data=res_df, errorbar='sd')
#         plt.savefig(os.path.join(folder, f'{init_name}{task}_{scores_names[i]}_line_per_budget.png'))  

    plt.figure()
    sns.lineplot(x='Budget', 