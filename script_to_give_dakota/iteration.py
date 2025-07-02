import os
import argparse
import subprocess
import yaml

import torch
import torch.nn.functional as F

from parse_mesh import extract_graph, get_correct_velocity_pressure
from foam_utils import save_pressure, save_U
from models import MeshGraphNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_original_scale(preds, metadata, graph):
    """
    Returns a clone of the prediction on the same scale as the original data
    """
    
    def get_base_field_name(field):
        """
        I call the fields something like 'velocity_x_solenoidal and want back only velocity_x
        """
        return '_'.join(field.split('_')[:2])
    preds = torch.clone(preds)
    means = torch.load('/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/train/processed/means.pt', weights_only=False)
    stds = torch.load('/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/train/processed/stds.pt', weights_only=False)
    for dim, field in enumerate(metadata['target_fields']):
        preds[:,dim] = preds[:,dim] * stds[field] + means[field]
        if 'solenoidal' in field:
            preds[:,dim] += graph[get_base_field_name(field)+'_potential'].squeeze()
    return preds


def load_and_predict(model_metadata_filepath, graph_filepath):

    with open(os.path.join(model_metadata_filepath, 'config.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)
    graph = extract_graph(graph_filepath, metadata['node_features'], metadata['edge_features'])
    model = MeshGraphNet(metadata, graph)
    model.load_state_dict(torch.load(os.path.join(model_metadata_filepath, 'models/epoch1999.pt'), map_location=device))
    model.eval()
    out = model(graph)
    out = get_original_scale(out, metadata, graph)
    velocity = out[:,:2]
    velocity = torch.cat([velocity, torch.zeros((velocity.size(0), 1))], dim=-1) # add z values
    pressure = out[:,2].unsqueeze(-1)
    return velocity, pressure, graph


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_and_metadata_folder', type=str,
                        default='/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/experiment19')
    parser.add_argument('--graph_filepath', type=str,
                        default='/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/test/raw/DSE-DACE100/workdir.1')
    args = parser.parse_args()

    velocity, pressure, graph = load_and_predict(args.model_and_metadata_folder, args.graph_filepath)
    
    velocity_true, pressure_true = get_correct_velocity_pressure(os.path.join(args.graph_filepath, '1000'))

    loss_velocity_x = F.mse_loss(velocity[:,0], velocity_true[:,0])
    loss_velocity_y = F.mse_loss(velocity[:,1], velocity_true[:,1])
    loss_pressure = F.mse_loss(pressure, pressure_true)
    with open(os.path.join(args.graph_filepath, 'loss.out'), 'w') as f:
        f.write(f"""loss velocity x: {loss_velocity_x}
                loss velocity y: {loss_velocity_y}
                loss pressure: {loss_pressure}""")
    print(f"""loss velocity x: {loss_velocity_x}
            loss velocity y: {loss_velocity_y}
            loss pressure: {loss_pressure}""")
    velocity_out_extruded = velocity[graph.nodes_of_each_type['out_extruded']]
    save_U(velocity, velocity_out_extruded, os.path.join(args.graph_filepath, '1001/Upred'))
    save_pressure(pressure, os.path.join(args.graph_filepath, '1001/ppred'))



if __name__=='__main__':
    main()








