import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
import random
import string
import torch
import torch.optim as optim
from torch import nn 

import sys
sys.path.append('./models')
sys.path.append('./dataset')
from pronet import ProNet
from gvpgnn import GVPNet
from segnn import SEGNN
from mace import MACE
from ec_dataset import ECdataset
from torch_geometric.data import DataLoader

import wandb
import warnings
warnings.filterwarnings("ignore")

def random_exp_name(prefix='exp', length=6):
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}_{suffix}"

### Args
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=9, help='Device to use')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in Dataloader')

### Data
parser.add_argument('--data_path', type=str, default='/mntc/yuhaoh/Data/Reaction-EC', help='path to load and process the data')

# data augmentation tricks
parser.add_argument('--mask', action='store_true', help='Random mask some node type')
parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
parser.add_argument('--deform', action='store_true', help='Deform node coords')
parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
parser.add_argument('--mask_aatype', type=float, default=0.1, help='Random mask aatype to 25(unknown:X) ratio')

### Training hyperparameter
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--lr_decay_step_size', type=int, default=50, help='Learning rate step size')
parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor') 
parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size during training')
parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size')

### Model
parser.add_argument('--model', type=str, default='ProNet', help='Choose from \'ProNet\'GVPNet\'SEGNN\'\MACE\'')
parser.add_argument('--level', type=str, default='backbone', help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
parser.add_argument('--num_blocks', type=int, default=3, help='Model layers')
parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
parser.add_argument('--out_channels', type=int, default=384, help='Number of classes, 1195 for the fold data, 384 for the ECdata')
parser.add_argument('--fix_dist', action='store_true')  
parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph') 
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')

# Integrating SCHull graph
parser.add_argument('--schull', type=eval, default=True, help='True | False')

# logging
parser.add_argument('--exp_name', type=str, default='lba', help='...')
parser.add_argument('--wandb', type=str, default='disabled', help='wandb mode')
parser.add_argument('--wandb_entity', type=str, default='utah-math-data-science', help='wandb entity')
parser.add_argument('--wandb_project', type=str, default='SCHull_on_EC', help='wandb project name')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--save_dir', type=str, default='/mntc/yuhaoh/Out/SCHull/EC', help='Number of GPUs to use')


args = parser.parse_args()
exp_name = random_exp_name(prefix=args.exp_name)
save_dir = os.path.join(args.save_dir, exp_name)
if not save_dir == "" and not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = nn.CrossEntropyLoss()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args, model, loader, optimizer, device):
    model.train()

    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader)):
        if args.mask:
            # random mask node aatype
            mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
            batch.x[:, 0][mask_indice] = 25
        if args.noise:
            # add gaussian noise to atom coords
            gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
            batch.coords_ca += gaussian_noise
        batch = batch.to(device)
                     
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))        
        function = batch.y
        functions.append(function)
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()        

    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()


def evaluation(args, model, loader, device):    
    model.eval()
    
    loss_accum = 0
    preds = []
    functions = []
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        # pred = model(batch)
        try:
            pred = model(batch) 
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): 
                print('\n forward error \n')
                raise(e)
            else:
                print('evaluation OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))
        
        function = batch.y
        functions.append(function)
        loss = criterion(pred, function)
        loss_accum += loss.item()    
            
    functions = torch.cat(functions, dim=0)
    preds = torch.cat(preds, dim=0)
    acc = torch.sum(preds==functions)/functions.shape[0]
    
    return loss_accum/(step + 1), acc.item()

    
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


    train_set = ECdataset(root=args.data_path, split='train')
    val_set = ECdataset(root=args.data_path, split='val')
    test_set = ECdataset(root=args.data_path, split='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    
    ##### set up model
    if args.model == 'ProNet':
        model = ProNet(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                cutoff=args.cutoff, dropout=args.dropout,
                data_augment_eachlayer=args.data_augment_eachlayer,
                euler_noise = args.euler_noise, level=args.level, schull=args.schull).to(device)
    elif args.model == 'GVPNet':
        model = GVPNet(schull=args.schull).to(device)
    elif args.model == 'SEGNN':
        model = SEGNN(cutoff=args.cutoff, dropout=args.dropout, 
                      in_dim=1, out_dim=1, 
                      hidden_features=args.hidden_channels, 
                      num_layers=args.num_blocks, schull=args.schull).to(device)
    elif args.model == 'MACE':
        model = MACE(r_max=args.cutoff,
                     num_layers=args.num_blocks,
                     mlp_dim=args.hidden_channels, 
                     out_channels=args.out_channels,
                     dropout=args.dropout, 
                     schull=args.schull).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    best_val_acc = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, model, train_loader, optimizer, device)
        val_loss, val_acc = evaluation(args, model, val_loader, device)
        print('Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.4f}, Val Loss: {:.6f}, Val Acc: {:.4f}'.format(
            epoch, train_loss, train_acc, val_loss, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not save_dir == "":
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            _, test_acc = evaluation(args, model, test_loader, device)
            test_acc_at_best_val = test_acc

        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc_at_best_val': test_acc_at_best_val
        })

if __name__ == "__main__":
    main()