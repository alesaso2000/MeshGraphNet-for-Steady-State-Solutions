import os
import argparse
import yaml

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import wandb


from define_dataset import MyOwnDataset
from models import MeshGraphNet
from utils import get_device



def setup_training(metadata):
    dataset = MyOwnDataset(metadata, split='train')
    print('size train dataset:', len(dataset))
    test_dataset = MyOwnDataset(metadata, split='test')
    print('size test dataset:', len(test_dataset))

    sample_graph = dataset.get(0)
    model = MeshGraphNet(metadata, sample_graph)

    optimizer = torch.optim.Adam(model.parameters(), lr=metadata['training_dynamics']['lr'])

    dataloader = DataLoader(dataset, 
                            batch_size=metadata['training_dynamics']['batch_size'], 
                            shuffle=True, 
                            num_workers=max(1, os.cpu_count()//4),
                            pin_memory=True,
                            prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, 
                            batch_size=metadata['training_dynamics']['batch_size'], 
                            shuffle=True, 
                            num_workers=max(1, os.cpu_count()//4),
                            pin_memory=True,
                            prefetch_factor=1)
    
    wandb.init(project='MeshGraphNet on cubotto', config=metadata, name=metadata['run'])
    return dataset, test_dataset, dataloader, test_dataloader, model, optimizer



def training_step(model, batch, optimizer):
    optimizer.zero_grad()
    preds = model(batch)
    losses = {'velocity_x': F.mse_loss(preds[:,0], batch.target[:,0]),
            'velocity_y': F.mse_loss(preds[:,1], batch.target[:,1]),
            'pressure': F.mse_loss(preds[:,2], batch.target[:,2])}
    losses['total'] = sum(losses.values()) / 3
    losses['total'].backward()
    optimizer.step()
    losses = {key: loss.item() for key,loss in losses.items()}
    return losses
    
    
@torch.no_grad()
def test(model, dataloader, device):

    def step(model, batch):
        preds = model(batch)    
        losses = {'velocity_x': F.mse_loss(preds[:,0], batch.target[:,0]),
                'velocity_y': F.mse_loss(preds[:,1], batch.target[:,1]),
                'pressure': F.mse_loss(preds[:,2], batch.target[:,2])}
        losses = {key: loss.item() for key,loss in losses.items()}
        return losses
    
    losses = {'velocity_x': 0, 'velocity_y': 0, 'pressure': 0}
    for batch in dataloader:
        batch = batch.to(device)
        loss = step(model, batch)
        losses = {key: l + loss[key] for key,l in losses.items()}
    losses = {key: l/len(dataloader) for key, l in losses.items()}
    losses['total'] = sum(losses.values()) / 3

    wandb.log({f'test_losses_{key}': l for key,l in losses.items()})



def train(metadata, dataloader, test_dataloader, model, optimizer, device):

    model = model.to(device)

    for e in range(metadata['training_dynamics']['n_epochs']):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            train_loss = training_step(model, batch, optimizer)
            wandb.log({f'train_losses_{key}': loss for key,loss in train_loss.items()})

        test(model, test_dataloader, device)

        torch.save(model.state_dict(), os.path.join(metadata['experiment_folder'], f'models/epoch{e}.pt'))
        
    return model


def main():
    parser = argparse.ArgumentParser(description="Parse metadata yaml file location.")
    parser.add_argument('--experiment_folder', type=str, required=True, 
                        help='the full file location like ./experiment1')
    args = parser.parse_args()

    with open(os.path.join(args.experiment_folder, 'config.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)

    metadata['experiment_folder'] = args.experiment_folder

    os.makedirs(os.path.join(metadata['experiment_folder'], 'models'))
    
    # ####### REMOVE FOR ACTUAL SCRIPT ###############
    # folder = 'experiment1'
    # with open(f'./{folder}/config.yaml', 'r') as f:
    #     metadata = yaml.safe_load(f)
    # metadata['experiment_folder'] = folder
    # os.makedirs(os.path.join(metadata['experiment_folder'], 'models'), exist_ok=True)
    # ################################################

    dataset, test_dataset, dataloader, test_dataloader, model, optimizer = setup_training(metadata)
    print('done setting up training')

    device = get_device(metadata)
    print(f'device: {device}')

    model = train(metadata, dataloader, test_dataloader, model, optimizer, device)

    wandb.finish()


if __name__=='__main__':
    main()
    