import os
import sys
sys.path.append('./dataset')
sys.path.append('./models')
import argparse
import random
import string
import json
import time
import wandb
import numpy as np
import torch

from collections import defaultdict
from functools import partial
from atom3d.util import metrics
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from lba_dataset import LBADataset
from pronet import ProNet
from gvpgnn import GVPNet

def random_exp_name(prefix='exp', length=6):
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}"

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', metavar='N', type=int, default=4, help='number of threads for loading data, default=4')
parser.add_argument('--dataset', type=str, default='lba')
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory. Should contain train, val, and test subdirectories.')

# Training
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_decay_step_size', type=int, default=80, help='Learning rate step size')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
parser.add_argument('--batch_size_eval', type=int, default=16, help='Batch size during training')

# Model
parser.add_argument('--model', type=str, default='ProNet', help='Choose from \'ProNet\'GVPNet\'')
parser.add_argument('--level', type=str, default='backbone', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
parser.add_argument('--num_blocks', type=int, default=3, help='Model layers')
parser.add_argument('--hidden_channels', type=int, default=256, help='Hidden dimension')
parser.add_argument('--out_channels', type=int, default=1, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
parser.add_argument('--fix_dist', action='store_true')  
parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')

## data augmentation tricks
parser.add_argument('--mask', type=eval, default=False, help='Random mask some node type')
parser.add_argument('--mask_aatype', type=float, default=1e-1, help='Random mask aatype to 25(unknown:X) ratio')
parser.add_argument('--noise', type=eval, default=False, help='Add Gaussian noise to node coords')
parser.add_argument('--noise_std', type=float, default=5e-2, help='Standard deviation of the Gaussian noise added to node coords')
parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')

# Integrating SCHull graph
parser.add_argument('--schull', type=eval, default=False, help='True | False')

# logging
parser.add_argument('--exp_name', type=str, default='lba', help='...')
parser.add_argument('--wandb', type=str, default='disabled', help='wandb mode')
parser.add_argument('--wandb_entity', type=str, default='utah-math-data-science', help='wandb entity')
parser.add_argument('--wandb_project', type=str, default='SCHull_on_LBA_02', help='wandb project name')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the model and logs')


args = parser.parse_args()
exp_name = random_exp_name(prefix=args.exp_name)
save_dir = os.path.join(args.save_dir, exp_name)
if not save_dir == "" and not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.MSELoss()

def get_datasets(data_path= None):
    trainset = LBADataset(os.path.join(data_path, 'train'))
    valset = LBADataset(os.path.join(data_path, 'val'))
    testset = LBADataset(os.path.join(data_path, 'test'))
    return trainset, valset, testset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

def get_model(args):
    if args.model == 'ProNet':
        model = ProNet(num_blocks=args.num_blocks, 
                       hidden_channels=args.hidden_channels, 
                       out_channels=args.out_channels,
                       cutoff=args.cutoff, dropout=args.dropout,
                       data_augment_eachlayer=args.data_augment_eachlayer,
                       euler_noise = args.euler_noise, level=args.level, 
                       schull=args.schull).to(device)
    elif args.model == 'GVPNet':
        model = GVPNet(num_blocks=args.num_blocks,
                       cutoff=args.cutoff,
                       out_channels=args.out_channels,
                       dropout=args.dropout, 
                       schull=args.schull).to(device)
    return model

def get_metrics():
    def _correlation(metric, targets, predict, ids=None, glob=True):
        targets = np.array(targets, dtype=np.float32)
        predict = np.array(predict, dtype=np.float32)
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])
    
    correlations = {
        'pearson': partial(_correlation, metrics.pearson),
        'kendall': partial(_correlation, metrics.kendall),
        'spearman': partial(_correlation, metrics.spearman),
        'rmse': partial(_correlation, metrics.rmse),
    }
    return {**correlations}


def train(args, model, loader, optimizer, device):
    model.train()
    train_loss = 0
    train_num = 0
    for _, batch in enumerate(tqdm(loader, disable=False)):
        if args.mask:
            # random mask node aatype
            mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
            batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.normal(mean=0.0, std=args.noise_std, size=batch.coords_ca.shape)
            batch.coords_ca += gaussian_noise
            if args.level != 'aminoacid':
                batch.coords_n += gaussian_noise
                batch.coords_c += gaussian_noise

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch).squeeze(dim=-1)
        label = batch.label
        batch_loss = criterion(pred, label)
        batch_loss = criterion(pred, label)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item() * batch.label.shape[0]
        train_num += batch.label.shape[0]
        
    return train_loss / train_num

def val(model, loader, device):
    model.eval()
    metrics = get_metrics()
    targets, predicts = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, disable=False)):
            batch = batch.to(device)
            pred = model(batch).squeeze(dim=-1)
            label = batch.label
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))
    val_dict = {} 
    for name, func in metrics.items():
        value = func(targets, predicts)
        val_dict[name] = value
    return val_dict

def test(model, loader, device):
    model.eval()
    metrics = get_metrics()
    targets, predicts = [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader, disable=False)):
            batch = batch.to(device)
            pred = model(batch).squeeze(dim=-1)
            label = batch.label
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))
    test_dict = {} 
    for name, func in metrics.items():
        value = func(targets, predicts)
        test_dict[name] = value
    return test_dict

def main():

    set_seed(args.seed)
    
    proj_name = '{}_seed{}_{}'.format(exp_name, args.seed, args.data_path)

    wand_save_dir = os.path.join(save_dir, 'wandb')
    if not os.path.exists(wand_save_dir):
        os.makedirs(wand_save_dir)
    wandb.init(entity = args.wandb_entity, 
               project = args.wandb_project, 
               mode = args.wandb,
               name = proj_name, 
               dir = wand_save_dir,
               config=args)
    
    # data and data loaders
    trainset, valset, testset = get_datasets(args.data_path)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(valset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(testset, batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers)
    # model and optimizer
    model = get_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                 weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=args.lr_decay_step_size, 
                                                gamma=args.lr_decay_factor)
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        t_start = time.perf_counter()
        train_loss = train(args, model, train_loader, optimizer, device)
        t_end_train = time.perf_counter()
        val_dict = val(model, val_loader, device)
        val_loss = val_dict['rmse'] 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, '{}_best.pth'.format(exp_name)))
            with open(os.path.join(save_dir, 'args.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
            test_results = test(model, test_loader, device)

        print('Epoch: {} | Train Loss: {:.6g} | Val Loss {:.6g} | Training Time: {:.4g}'.format(epoch, train_loss, val_loss, t_end_train - t_start))
        wandb.log({'epoch': epoch, 
                   'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   'best_val_loss': best_val_loss,
                   'test_rmse_at_best_val': test_results['rmse'],
                   'test_pearson_at_best_val': test_results['pearson'],
                   'test_kendall_at_best_val': test_results['kendall'],
                   'test_spearman_at_best_val': test_results['spearman'],})

        scheduler.step() 


if __name__ == '__main__':
    main()