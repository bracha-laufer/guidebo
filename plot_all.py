from utils import plot_res
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sns.set(style="whitegrid",font_scale=1.3)


import os
import json

datasets = ['fariness']


def main():
    tasks = ['vae', 'fairness', 'etc']
    task_names = ['vae', 'fairness', 'etc']

    seeds = [0,1,2,3,4]

    fig_type = 'all'
    figs_folder = 'figures' 
    gamma =[0.01]
    gamma_proposed = [0.01]
    n_trials = 20
    init_names = ['all']
    init_names_proposed = ['all']
    if fig_type == 'all':
        init_fig = ''
        selected_methods = ["Uniform", "Random", "EHVI", "HVI", "ParEGO"]       
    elif fig_type == 'upper':
        init_fig = 'upper_'
        selected_methods = ["Upper Limit"]
   
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)
    
    
    df_all_tasks = []
    for task_name, task, init_name, init_name_proposed in zip(task_names, tasks, init_names, init_names_proposed):
        res_name =  f'results_{task}'
        df_list = []

        for seed in seeds:
            directory = f'results_{task}/seed_{seed}/{init_name}_dprime_s_{gamma}_{gamma}_e_{gamma}_{gamma}_trials_{n_trials}'     
            df_path = os.path.join(directory, res_name + '.csv')
            df = pd.read_csv(df_path)
            df = df[df['Method'].isin(selected_methods)]
            
            directory = f'results_{task}/seed_{seed}/{init_name_proposed}_dprime_s_{gamma_proposed}_{gamma_proposed}_e_{gamma_proposed}_{gamma_proposed}_trials_{n_trials}'     
            df_path = os.path.join(directory, res_name + '.csv')
            df_proposed = pd.read_csv(df_path)
            df_proposed = df_proposed[df_proposed['Method']=='Proposed']
            
        
            df_list.append(pd.concat([df_proposed, df],ignore_index=True))
            config_path = os.path.join(directory, 'config.json')

        with open(config_path) as cf_file:
            config = json.loads(cf_file.read()) 

        df = pd.concat(df_list,ignore_index=True)
        df['Method'] = df['Method'].replace('Proposed', 'GuideBO')
        filtered_df = df
        
        #filtered_df = df[df['Method'].isin(selected_methods)]
        #filtered_df = pd.concat([df[df['Method']==m] for m in selected_methods])
        

        #filtered_df = filtered_df.sort_values(by=['Method'])
        plot_res(figs_folder, task_name, config["objs_names"], filtered_df, init_fig)
        grouped_sr = filtered_df.groupby([f'$\\alpha_{1}$', 'Method']).mean()
        grouped_df = grouped_sr.reset_index()
        grouped_df = grouped_df.round(3)
        grouped_df["Rank"] = grouped_df.groupby(f'$\\alpha_{1}$')[config["objs_names"][-1]].rank(method='min')
        print(grouped_df)
        print(grouped_df.groupby('Method')["Rank"].mean())
        df_all_tasks.append(grouped_df)

    df_all = pd.concat(df_all_tasks,ignore_index=True)
    selected_methods.append('GuideBO')
    print(df_all.groupby('Method')["Rank"].mean())
    print(df_all.groupby(['Method',"Rank"])["Rank"].count())
    rank_count = df_all.groupby(['Method',"Rank"])["Rank"].count().reset_index(name="count")
    rank_count = rank_count.reset_index()
    rank_count["count"] = rank_count["count"].round().astype(int)
    rank_count["Rank"] = rank_count["Rank"].round().astype(int)
    print(rank_count)
    table = pd.pivot_table(rank_count, values='count', index='Method', columns=['Rank'], fill_value=0.0)
    table['Avg. Rank'] = df_all.groupby('Method')["Rank"].mean().round(1)
    for i in range(len(selected_methods)):
        table[i+1] = table[i+1].round().astype(int)
    table = table.sort_values(by=['Avg. Rank'])
    print(table)
    fig, ax = plt.subplots()
    table1 = table.copy()
    table1['Avg. Rank'] = float('nan')
    ax = sns.heatmap(table1, annot=True, cmap=sns.cubehelix_palette(as_cmap=True)
, linewidth=.5, cbar_kws = dict(location="top", shrink= 0.5))
    #plt.colorbar(aspect=20)
    table2 = table.copy()
    table2.loc[:, table2.columns != 'Avg. Rank'] = float('nan')
    ax = sns.heatmap(table2, annot=True, linewidth=2, cmap =sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
, cbar_kws = dict(shrink= 0.6))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.labelpad = -30

  
    ax.add_patch(
    patches.Rectangle(
         (6, 0),
         1.0,
         6.0,
         edgecolor='white',
         fill=False,
         lw=5
     ))

    plt.savefig(os.path.join(figs_folder, f'ranks_{fig_type}.png'))  



if __name__ == '__main__':
    main()      
