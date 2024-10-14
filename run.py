import subprocess
import concurrent.futures
import json
import argparse
from main import main 


task_config_dic = {'fairness': 'configs/fariness_mixup.json',
                    'vae': 'configs/vae.json',
                    'etc': 'configs/etc.json'}

def run():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--tasks', type=str, nargs='+', help='tasks to run')
    parser.add_argument('--seeds', type=int, nargs='+', help='seeds to run')
    args = parser.parse_args()

    multi_thread = False
    seeds = args.seeds
    max_workers = 5
    task_names = args.tasks 
    configs_to_run = [task_config_dic[task] for task in task_names] 
    methods = ["Proposed", "Uniform", "Random", "EHVI", "HVI", "ParEGO"]
    
    config_bo_file = 'configs/bo.json'  
    with open(config_bo_file) as cf_file:
        config_bo = json.loads(cf_file.read())
    
    tasks = []
    for seed in seeds:
        for config_task_file in configs_to_run:
            with open(config_task_file) as cf_file:
                config= json.loads(cf_file.read())       

            config.update(config_bo) 
            tasks.append((config.copy(), methods, seed))
    
    if multi_thread:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                 for task in tasks:
                    result = executor.submit(lambda p: main(*p), task)
                    try:
                        result = future.result()
                    except:
                        print('failed')

            print('All done')
    else:
        for task in tasks:
            print(f'seed = {task[2]}')
            main(*task)


if __name__ == '__main__':
    run()        

        
