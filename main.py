import pandas as pd
import numpy as np
import lib_oversampling as lo
import lib_completion as lc
import ddpm
import data_utils as du
import os
import torch
import argparse

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--task-name', type=str, default='oversampling')
parser.add_argument('--dataset-name', type=str, default='default')

parser.add_argument('--diffuser-dim', nargs='+', type=int, default=(512, 1024, 1024, 512))
parser.add_argument('--diffuser-lr', type=float, default=0.0018)
parser.add_argument('--diffuser-steps', type=int, default=30000)
parser.add_argument('--diffuser-bs', type=int, default=4096)
parser.add_argument('--diffuser-timesteps', type=int, default=1000)

parser.add_argument('--controller-dim', nargs='+', type=int, default=(512, 512))
parser.add_argument('--controller-lr', type=float, default=0.001)
parser.add_argument('--controller-steps', type=int, default=10000)
parser.add_argument('--controller-bs', type=int, default=512)

parser.add_argument('--device', type=int, default=1)
parser.add_argument('--scale-factor', type=float, default=8.0)
parser.add_argument('--save-name', type=str, default='output')
args = parser.parse_args()

save_dir = os.path.join('expdir', args.save_name)
os.makedirs(save_dir, exist_ok=True)
device = torch.device(f'cuda:{args.device}')

if args.task_name == 'oversampling':
    config = du.load_json(f'datasets/minority_class_oversampling/dataset_info.json')[args.dataset_name]
    train_data = pd.read_csv(f'datasets/minority_class_oversampling/{args.dataset_name}_train.csv')
    test_data = pd.read_csv(f'datasets/minority_class_oversampling/{args.dataset_name}_test.csv')
    
    all_data = pd.concat((train_data, test_data))
    data_wrapper, label_wrapper = lo.data_preprocessing(all_data, config['label'], save_dir)

    ''' diffuser training '''
    train_x = data_wrapper.transform(train_data)
    lo.diffuser_training(train_x = train_x, 
                        save_path = os.path.join(save_dir, 'diffuser.pt'), 
                        device=device, 
                        d_hidden=args.diffuser_dim, 
                        num_timesteps=args.diffuser_timesteps, 
                        epochs=args.diffuser_steps, 
                        lr=args.diffuser_lr, 
                        drop_out=0.0, 
                        bs=args.diffuser_bs)

    ''' controller training '''
    diffuser = torch.load(os.path.join(save_dir, 'diffuser.pt'))
    label = config['label']
    n_classes = len(pd.unique(train_data[label]))
    train_x = data_wrapper.transform(train_data)
    train_y = label_wrapper.transform(train_data[[label]])

    lo.controller_training(train_x=train_x,
                        train_y=train_y, 
                        diffuser=diffuser, 
                        save_path=os.path.join(save_dir, 'controller.pt'), 
                        device=device, 
                        n_classes=n_classes, 
                        lr=args.controller_lr, 
                        d_hidden=args.controller_dim, 
                        steps=args.controller_steps, 
                        drop_out=0.0, 
                        bs=args.controller_bs)

    ''' oversampling '''
    diffuser = torch.load(os.path.join(save_dir, 'diffuser.pt'))
    controller = torch.load(os.path.join(save_dir, 'controller.pt'))
    
    sample_data = []
    unique_labels, counts = np.unique(train_y, return_counts=True)
    label_ratios = counts / counts.sum()
    
    for i, label in enumerate(unique_labels):
        total_samples = config['n_samples'][0]
        n_samples = int(total_samples * label_ratios[i])
        samples = lo.oversampling(n_samples, controller, diffuser, label, device, n_classes, args.scale_factor)
        sample_data.append(samples)

    sample_data = torch.cat(sample_data, dim=0)
    sample_data = sample_data.cpu().numpy()
    sample_data = data_wrapper.Reverse(sample_data)
    pd.DataFrame(sample_data).to_csv(os.path.join(save_dir, 'generated_data.csv'), index=None)  





