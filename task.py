def from_task(task_name):
    print(task_name)
    if task_name=='fairness':
       from fairness.fairness_main import train, evaluate_path, init_configs, get_eff_sizes
    if task_name=='vae':
       from vae.vae_main import train, evaluate_path, init_configs, get_eff_sizes
    if task_name=='etc':
       from etc.etc_main import train, evaluate_path, init_configs, get_eff_sizes
   
    return train, evaluate_path, init_configs, get_eff_sizes
