# @Time   : 2020/7/20
# @Author : sahanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import sys, os
import shutil
from recbole.quick_start import run_recbole


def recbole_run(model, dataset):

    

    os.chdir(sys.path[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=f'{model}', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default=f'{dataset}', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default='current_bpr.yaml', help='config files')
    
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

def check_params(log_files, config_dict):

    found = False
    for log_file in log_files:
        
        # read all lines
        with open(log_file, 'r') as fin:
            lines = fin.readlines()

        # check if the log is complete (contains best valid and test result at the end of the log)
        if not ("best valid : OrderedDict" in lines[-2] and "test result: OrderedDict" in lines[-1]):
            continue

        # start search
        found_here = True 

        for line in lines:
            line = line.strip()

            if ' = ' not in line:
                continue

            key_line, value_line = line.split(' = ')

            for key_dict in config_dict:
                if key_dict == key_line:
                    if not value_line == str(config_dict[key_dict]):
                        found_here = False
                        break

            if not found_here:
                break

        found = found_here

        if found:
            return found

    return found




if __name__ == '__main__':

    '''
    weight_decay choice [0.0001, 0.001, 0.01]
    '''

    model = 'BPR'
    dataset = 'mm_ml1m'

    # create folder if not existing
    os.makedirs('log', exist_ok=True)
    os.makedirs(f'log/{model}/', exist_ok=True)
    
    for weight_decay in [0.0001, 0.001, 0.01]:

        # check if this exists in the log
        # i need to check:
        # - dataset
        # - model
        # - param values

        # i use a dict str -> value to give the function these config

        config_dict = {
            'weight_decay': weight_decay

        }

        logs = [f'log/{model}/{log_file}' for log_file in os.listdir(f'log/{model}/')]
        
        # this run must be started
        if not check_params(logs, config_dict):

            # run the model with this setting
            shutil.copy('confs/conf_bpr.yaml', 'current_bpr.yaml')
            with open('current_bpr.yaml', 'a') as f:
                f.write(f'\nweight_decay: {weight_decay}')
            recbole_run(model, dataset)
            os.remove('current_bpr.yaml')
            
    