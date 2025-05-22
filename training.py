import os
import argparse
import yaml

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torch_geometric.loader import DataLoader
import wandb


from define_dataset import MyOwnDataset
from models import MeshGraphNet
from utils import get_device, get_original_scale, get_base_field_name



def setup_training(metadata):
    dataset = MyOwnDataset(metadata, split='train')
    print('size train dataset:', len(dataset))
    test_dataset = MyOwnDataset(metadata, split='test')
    print('size test dataset:', len(test_dataset))

    sample_graph = dataset.get(0)
    model = MeshGraphNet(metadata, sample_graph)

    optimizer = Adam(model.parameters(), lr=metadata['training_dynamics']['lr'])
    scheduler = StepLR(optimizer, step_size=metadata['training_dynamics']['scheduler_step_size'], 
                       gamma=metadata['training_dynamics']['scheduler_gamma'])

    dataloader = DataLoader(dataset, 
                            batch_size=metadata['training_dynamics']['batch_size'], 
                            shuffle=True, 
                            num_workers=max(1, os.cpu_count()//4),
                            pin_memory=True,
                            prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=metadata['training_dynamics']['batch_size'], 
                            shuffle=False, 
                            num_workers=max(1, os.cpu_count()//4),
                            pin_memory=True,
                            prefetch_factor=1)
    
    wandb.init(project='MeshGraphNet on cubotto', config=metadata, name=metadata['run'])
    return dataset, test_dataset, dataloader, test_dataloader, model, optimizer, scheduler



def training_step(model, batch, optimizer, metadata):
    optimizer.zero_grad()
    preds = model(batch)
    losses = {}
    for dim, field in enumerate(metadata['target_fields']):
        losses[field] = F.mse_loss(preds[:,dim], batch.target[:,dim])
    losses['total'] = sum(losses.values()) / len(losses.keys())
    losses['total'].backward()
    optimizer.step()
    losses = {key: loss.item() for key,loss in losses.items()}
    return losses
    
    
@torch.no_grad()
def test(model, dataloader, device, means, stds, metadata):

    def step(model, batch, means, stds):
        preds = model(batch)    
        losses = {}
        for dim, field in enumerate(metadata['target_fields']):
            losses[field] = F.mse_loss(preds[:,dim], batch.target[:,dim]).item()
        
        preds = get_original_scale(preds, metadata, means, stds, batch)
        losses_original_scale = {}
        for dim, field in enumerate(metadata['target_fields']):
            # batch.velocity_x etc are not normalized, while batch.target is
            field = get_base_field_name(field)  
            losses_original_scale[field] = F.mse_loss(preds[:,dim], batch[field].squeeze()).item()
        return losses, losses_original_scale
    
    losses = {field: 0 for field in metadata['target_fields']}
    losses_original_scale = {get_base_field_name(field): 0 for field in metadata['target_fields']}
    for batch in dataloader:
        batch = batch.to(device)
        loss, loss_orig = step(model, batch, means, stds)
        losses = {key: l + loss[key] for key,l in losses.items()}
        losses_original_scale = {key: l + loss_orig[key] for key, l in losses_original_scale.items()}
    losses = {key: l/len(dataloader) for key, l in losses.items()}
    losses['total'] = sum(losses.values()) / len(losses.keys())
    losses_original_scale = {key: l/len(dataloader) for key, l in losses_original_scale.items()}
    losses_original_scale['total'] = sum(losses_original_scale.values()) / len(losses_original_scale.keys())

    wandb.log({f'test_losses_{key}': l for key,l in losses.items()})
    wandb.log({f'test_losses_orig_scale_{key}': l for key, l in losses_original_scale.items()})



def train(metadata, dataloader, test_dataloader, model, optimizer, scheduler, device, dataset):
    model = model.to(device)
    means = {key: val.to(device) for key, val in dataset.mean.items()}
    stds = {key: val.to(device) for key, val in dataset.std.items()}

    for e in range(metadata['training_dynamics']['n_epochs']):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            train_loss = training_step(model, batch, optimizer, metadata)
            if i%metadata['training_dynamics']['log_every'] == 0:
                wandb.log({f'train_losses_{key}': loss for key,loss in train_loss.items()})

        test(model, test_dataloader, device, means, stds, metadata)

        torch.save(model.state_dict(), os.path.join(metadata['experiment_folder'], f'models/epoch{e}.pt'))

        scheduler.step()
        
    return model


def main():
    parser = argparse.ArgumentParser(description="Parse metadata yaml file location.")
    parser.add_argument('--experiment_folder', type=str, required=True, 
                        help='the full file location; something like ./experiment1')
    args = parser.parse_args()

    with open(os.path.join(args.experiment_folder, 'config.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)

    metadata['experiment_folder'] = args.experiment_folder

    os.makedirs(os.path.join(metadata['experiment_folder'], 'models'))
    
    # ####### REMOVE FOR ACTUAL SCRIPT ###############
    # folder = 'experiment3'
    # with open(f'./{folder}/config.yaml', 'r') as f:
    #     metadata = yaml.safe_load(f)
    # metadata['experiment_folder'] = folder
    # os.makedirs(os.path.join(metadata['experiment_folder'], 'models'), exist_ok=True)
    # ################################################

    dataset, test_dataset, dataloader, test_dataloader, model, optimizer, scheduler = setup_training(metadata)
    print('done setting up training')

    # device = get_device(metadata)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    model = train(metadata, dataloader, test_dataloader, model, optimizer, scheduler, device, dataset)

    wandb.finish()


if __name__=='__main__':
    main()
    