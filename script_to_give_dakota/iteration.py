import os
import argparse
import subprocess
import yaml

import torch
import torch.nn.functional as F

from parse_mesh import extract_graph
from foam_utils import create_boundary_field, save_pressure, save_U
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
        # print(f'loss {field}', F.mse_loss(preds[:,dim].unsqueeze(-1), graph[field]))
    return preds


def load_and_predict(model_metadata_filepath, graph_filepath):

    with open(os.path.join(model_metadata_filepath, 'config.yaml'), 'r') as f:
        metadata = yaml.safe_load(f)
    graph = extract_graph(graph_filepath, metadata['node_features'], metadata['edge_features'])
    model = MeshGraphNet(metadata, graph)
    model.load_state_dict(torch.load(os.path.join(model_metadata_filepath, 'models/epoch1999.pt'), map_location=device))

    out = model(graph)
    out = get_original_scale(out, metadata, graph)
    velocity = out[:,:2]
    velocity = torch.cat([velocity, torch.zeros((velocity.size(0), 1))], dim=-1) # add z values
    pressure = out[:,2].unsqueeze(-1)
    return velocity, pressure, graph


# def get_boundary(velocity, pressure, graph):

#     # create a dictionary with keys  prof, free, in, out and values the corresponding 
#     velocity_boundary = {}
#     pressure_boundary = {}
#     # prof_extruded
#     nodes_idx = graph.nodes_of_each_type['prof_extruded']
#     velocity_boundary['prof_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['prof_extruded'] = pressure[nodes_idx]
#     # free_extruded   
#     # TODO: Check with damiano about boundary conditions
#     nodes_idx = graph.nodes_of_each_type['free_extruded']
#     velocity_boundary['free_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['free_extruded'] = torch.zeros_like(pressure[nodes_idx])
#     # in_extruded  
#     # TODO: Check with damiano about boundary conditions
#     nodes_idx = graph.nodes_of_each_type['in_extruded']
#     velocity_boundary['in_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['in_extruded'] = torch.zeros_like(pressure[nodes_idx])
#     # out_extruded
#     nodes_idx = graph.nodes_of_each_type['out_extruded']
#     velocity_boundary['out_extruded'] = velocity[nodes_idx]
#     pressure_boundary['out_extruded'] = pressure[nodes_idx]

#     return velocity_boundary, pressure_boundary


# def get_boundary(velocity, pressure, graph):

#     # create a dictionary with keys  prof, free, in, out and values the corresponding 
#     velocity_boundary = {}
#     pressure_boundary = {}
#     # prof_extruded
#     nodes_idx = graph.nodes_of_each_type['prof_extruded']
#     velocity_boundary['prof_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['prof_extruded'] = pressure[nodes_idx]
#     # free_extruded   
#     # TODO: Check with damiano about boundary conditions
#     nodes_idx = graph.nodes_of_each_type['free_extruded']
#     velocity_boundary['free_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['free_extruded'] = torch.zeros_like(pressure[nodes_idx])
#     # in_extruded  
#     # TODO: Check with damiano about boundary conditions
#     nodes_idx = graph.nodes_of_each_type['in_extruded']
#     velocity_boundary['in_extruded'] = torch.zeros_like(velocity[nodes_idx])
#     pressure_boundary['in_extruded'] = torch.zeros_like(pressure[nodes_idx])
#     # out_extruded
#     nodes_idx = graph.nodes_of_each_type['out_extruded']
#     velocity_boundary['out_extruded'] = velocity[nodes_idx]
#     pressure_boundary['out_extruded'] = pressure[nodes_idx]

#     return velocity_boundary, pressure_boundary



# def save(velocity, pressure, velocity_boundary, pressure_boundary):

#     save_foam_field(
#         ofpp_internal=velocity,
#         ofpp_boundary=velocity_boundary,
#         object_field='U',
#         output_file='./Upred(delete)',
#         field_class='volVectorField', 
#         field_location=str(1000), 
#         field_dimensions='[0 1 -1 0 0 0 0]'
#     )

#     save_foam_field(
#         ofpp_internal=pressure, 
#         ofpp_boundary=pressure_boundary, 
#         object_field='ppred', 
#         output_file='./ppred(delete)', 
#         field_class='volScalarField', 
#         field_location=str(1000), 
#         field_dimensions='[0 2 -2 0 0 0 0]'
#     )


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_and_metadata_folder', type=str,
                        default='/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/experiment13')
    parser.add_argument('--graph_filepath', type=str,
                        default='/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/test/raw/DSE-DACE100/workdir.1')
    args = parser.parse_args()

    velocity, pressure, graph = load_and_predict(args.model_and_metadata_folder, args.graph_filepath)

    velocity_out_extruded = velocity[graph.nodes_of_each_type['out_extruded']]
    save_U(velocity, velocity_out_extruded, './U_delete')
    save_pressure(pressure, './pressure_delete')




    # velocity_boundary, pressure_boundary = get_boundary(velocity, pressure, graph)
    # velocity_boundary = torch.cat([v for v in velocity_boundary.values()], dim=0)
    # pressure_boundary = torch.cat([v for v in pressure_boundary.values()], dim=0)
    # velocity_boundary = create_boundary_field(
    #     '/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/test/raw/DSE-DACE100/workdir.1/1000/C', 
    #     velocity_boundary)
    # pressure_boundary = create_boundary_field(
    #     '/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/test/raw/DSE-DACE100/workdir.1/1000/C', 
    #     pressure_boundary)
    # save(velocity, pressure, velocity_boundary, pressure_boundary)
    



if __name__=='__main__':
    main()




