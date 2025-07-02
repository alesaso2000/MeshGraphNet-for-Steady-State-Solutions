import os
import re
import copy
from typing import List, Union
import pickle

import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from utils_potential import body_panels, sol_pot, mk_coeff_xy
from utils import get_ordered_body_coords


class MyOwnDataset(Dataset):

    def __init__(self, metadata, root='./data', split='train'): 
        self.metadata = metadata  
        self.split = split 
        self.which_folders = [200, 500] if split=='train' else [100]          
        super().__init__(root=os.path.join(root, split))   
        
        stats_dir = os.path.join(root, 'train/processed')
        self.mean = torch.load(os.path.join(stats_dir, "means.pt"), weights_only=False)
        self.std = torch.load(os.path.join(stats_dir, "stds.pt"), weights_only=False)

    @property
    def raw_file_names(self):
        return [f'DSE-DACE{N}/workdir.{i}' 
                for N in self.which_folders for i in range(1,N+1)]
    
    @property
    def processed_file_names(self):
        if os.path.exists(os.path.join(self.processed_dir, 'filenames.pkl')):
            with open(os.path.join(self.processed_dir, 'filenames.pkl'), 'rb') as f:
                filenames = pickle.load(f)
            return filenames
        else:
            return ['have_not_processed_yet.pt']
    

    def process(self):
        def extract_polymesh(file_path: str):  #file path is ..../polyMesh
            """
            Extracts the data about the mesh which is found in the polymesh folder. 
            """
            def extract_topology():
                def extract(what: str, n_line: int, start_line: int, to_type):
                    with open(os.path.join(file_path, what), 'r') as f:
                        lines = f.read().splitlines()
                    n = int(lines[n_line])
                    return torch.tensor([to_type(l) for l in lines[start_line: start_line+n]])

                owners = extract('owner', 18, 20, int)
                neighbours = extract('neighbour', 18, 20, int)
            
                return owners, neighbours
        
            def extract_node_type(owners: torch.Tensor):
                def parse(lines):
                    patch_pattern = re.compile(
                        r'(\w+)\s*\{\s*[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);',
                        re.MULTILINE | re.DOTALL
                    )

                    patches = {}
                    for match in patch_pattern.finditer(lines):
                        patch_name = match.group(1)
                        nFaces = int(match.group(2))
                        startFace = int(match.group(3))
                        patches[patch_name] = {
                            'nFaces': nFaces,
                            'startFace': startFace
                        }
                    del patches['defaultFaces']
                    return patches

                with open(os.path.join(file_path, 'boundary'), 'r') as f:
                    lines = f.read()

                patches = parse(lines)
                # the types are in order prof, free, in, out
                nodes_of_each_type = {key: owners[patches[key]['startFace']:patches[key]['startFace']+patches[key]['nFaces']] 
                                      for key in patches.keys()}

                node_type_one_hot = torch.zeros(size=(len(owners.unique()), len(patches.keys())+1))
                for i,key in enumerate(patches.keys()):
                    node_type_one_hot[nodes_of_each_type[key],i] = 1

                node_type_one_hot[node_type_one_hot.sum(axis=1)==0,-1] = 1

                return node_type_one_hot


            def extract_points():
                with open(os.path.join(file_path, 'points'), 'r') as f:
                    lines = f.read().splitlines()
                n_points = int(lines[17])
                points = [[float(p) for p in point[1:-1].split()] for point in lines[19:19+n_points]]
                points = torch.tensor(points)
                return points
            

            def extract_faces():
                with open(os.path.join(file_path, 'faces'), 'r') as f:
                    lines = f.read().splitlines()
                n_faces = int(lines[17])
                faces = [[int(f) for f in face[2:-1].split()] for face in lines[19:19+n_faces]]   # can't make it a tensor as some faces have 4 and other 3 points
                return faces  
                
            
            owners, neighbours = extract_topology()
            node_type_one_hot = extract_node_type(owners)
            points = extract_points()
            faces = extract_faces()

            return owners, neighbours, points, faces, node_type_one_hot
        

        def make_edges(owners, neighbours, centers):
            edge_index = torch.stack([neighbours, owners[:len(neighbours)]])
            edge_index = torch.cat([edge_index, edge_index.flip(dims=[0])], dim=1)  # to undirected
            distance_vector = centers[edge_index[0]] - centers[edge_index[1]]
            distance_magnitude = torch.linalg.norm(distance_vector, dim=-1, keepdim=True)
            return edge_index, distance_vector, distance_magnitude
        

        def get_distance_from_object(node_type_one_hot, centers):
            obj_center = (centers[node_type_one_hot[:,0]==1]).mean(dim=0)
            distance_vector = centers - obj_center
            distance_magnitude = torch.linalg.norm(distance_vector, dim=1, keepdim=True)
            return distance_vector, distance_magnitude


        def extract_fields(file_path: str):   # file_path is .../1000
            """
            From the file of the final timestep, I will extract the fields (pressure and velocity) and their time average
            """
            def extract(what: str, n_line=19, start_line=21):
                with open(os.path.join(file_path, what), 'r') as f:
                    lines = f.read().splitlines()
                n = int(lines[n_line])
                return torch.tensor([[float(l) for l in line[1:-1].split()[:2]] for line in lines[start_line:start_line+n]])
            
            def extract_pressure(which: str, n_line=19, start_line=21):
                with open(os.path.join(file_path, which), 'r') as f:
                    lines = f.read().splitlines()
                n = int(lines[n_line])
                return torch.tensor([float(l) for l in lines[start_line: start_line+n]]).unsqueeze(-1)
            
            def extract_body_coords():
                with open(os.path.join(file_path, 'C'), 'r') as f:
                    lines = f.read().splitlines()
                prof_extruded_line = lines.index('    prof_extruded')
                n = int(lines[prof_extruded_line + 4])
                return torch.tensor([[float(c) for c in line[1:-1].split()[:2]] for line in lines[prof_extruded_line+6 : prof_extruded_line+6+n]])
                 

            velocity = extract('U')
            velocity_mean = extract('UMean')
            centers = extract('C')
            pressure = extract_pressure('p')
            pressure_mean = extract_pressure('pMean')
            body_coords = extract_body_coords()

            return velocity, velocity_mean, pressure, pressure_mean, centers, body_coords
        

        def extract_bspline_points(file_path):
            with open(file_path) as f:
                line = f.read().split()
            points = [float(p) for p in line]
            return torch.tensor(points) # west_x, east_x, north_y, south_y
        

        def check_orderability(graph):
            """"
            Checks that the nodes on the object are easily orderable (i.e. the tips are not higher or lower than the center)
            """
            check = graph.body_coords[:,1].min() >= graph.bspline_points[-1] and graph.body_coords[:,1].max() <= graph.bspline_points[-2]
            return check
        

        def extract_graph(N: int, i: int):
            """
            Extracts the data from the OPENFoam output and makes it into a graph.
            Since the simulation is finite volume, I will construct and use the graph which connects the cell centers (of adjecent cells),
            not the mesh that has as edges the edge of the cell 
            """
            owners, neighbours, points, faces, node_type_one_hot = extract_polymesh(os.path.join(self.raw_dir, f'DSE-DACE{N}/workdir.{i}/constant/polyMesh'))
            velocity, velocity_mean, pressure, pressure_mean, centers, body_coords = extract_fields(os.path.join(self.raw_dir, f'DSE-DACE{N}/workdir.{i}/1000'))
            bspline_points = extract_bspline_points(os.path.join(self.raw_dir, f'DSE-DACE{N}/workdir.{i}/input.txt'))
            edge_index, distance_vector, distance_magnitude = make_edges(owners, neighbours, centers)
            distance_from_obj_vec, distance_from_obj_mag = get_distance_from_object(node_type_one_hot, centers)

            return Data(**{'node_type_one_hot': node_type_one_hot, 'edge_index': edge_index, 
                        'distance_vector': distance_vector, 'distance_magnitude': distance_magnitude,
                        'velocity_x': velocity_mean[:,0].unsqueeze(-1), 'velocity_y': velocity_mean[:,1].unsqueeze(-1),
                        'pressure': pressure_mean,
                        'pos': centers, 'bspline_points': bspline_points,
                        'distance_from_obj_vec': distance_from_obj_vec, 
                        'distance_from_obj_mag': distance_from_obj_mag,
                        'body_coords': body_coords})
        

        def get_potential_solution(graph):
            ordered_body_coords = get_ordered_body_coords(graph)
            ordered_body_coords = torch.cat([ordered_body_coords, ordered_body_coords[0].unsqueeze(0)], dim=0)
            xycpan, xpan, ypan, thpan, nor = body_panels(ordered_body_coords.numpy()) # (N,2)
            sigma_x, sigma_y = sol_pot(xycpan, xpan, ypan, thpan, nor)
            Ax_field, Ay_field = mk_coeff_xy(graph.pos.numpy(), thpan, xpan, ypan)  
            u_pot_field = np.matmul(Ax_field, sigma_x) + np.matmul(Ax_field, 0*sigma_y) + 1
            u_pot_field = torch.from_numpy(u_pot_field).float()
            v_pot_field = np.matmul(Ay_field, sigma_x) + np.matmul(Ay_field, 0*sigma_y) + 0
            v_pot_field = torch.from_numpy(v_pot_field).float()
            potential_solution = torch.cat([u_pot_field, v_pot_field], dim=1)
            return potential_solution
        

        def get_fourier_feature(graph):
            frequencies = torch.tensor([2, 4, 8, 16, 32]) * torch.pi  # shape (5,)
            minmaxed_positions = (graph.pos - graph.pos.min(dim=0).values) / (graph.pos.max(dim=0).values - graph.pos.min(dim=0).values) # shape (N,2)
            arg = minmaxed_positions.unsqueeze(2) * frequencies  # shape(N,2,4)
            sines = torch.sin(arg).reshape(graph.pos.shape[0], graph.pos.shape[1]*frequencies.shape[0])  # (points, coordinate, freq) -> (points, sin([x(freq[0]), y(freq[0]),..., x(freq[4]), y(freq[4])]))
            cosines = torch.cos(arg).reshape(graph.pos.shape[0], graph.pos.shape[1]*frequencies.shape[0]) 
            return torch.cat([sines, cosines], dim=1)


        mean = {'distance_vector': 0, 'distance_magnitude': 0, 
               'velocity_x': 0, 'velocity_y': 0, 'pressure': 0,
               'distance_from_obj_vec': 0,
               'distance_from_obj_mag': 0,
               'velocity_x_potential': 0,
               'velocity_y_potential': 0,
               'velocity_x_solenoidal': 0,
               'velocity_y_solenoidal': 0,
               'pos': 0,
               'fourier_feature': 0}
        mean2 = copy.deepcopy(mean)
        
        filenames = []
        for N in self.which_folders:
            for i in range(1, N+1):
                # extract and save
                graph = extract_graph(N, i)
                if not check_orderability(graph):
                    print('discarded', N, i)
                    continue
                graph.fourier_feature = get_fourier_feature(graph)
                potential_solution = get_potential_solution(graph)
                graph.velocity_x_potential = potential_solution[:,0].unsqueeze(-1)
                graph.velocity_y_potential = potential_solution[:,1].unsqueeze(-1)
                graph.velocity_x_solenoidal = graph.velocity_x - graph.velocity_x_potential
                graph.velocity_y_solenoidal = graph.velocity_y - graph.velocity_y_potential
                torch.save(graph, os.path.join(self.processed_dir, f'DSE-DACE{N}_workdir{i}.pt'))
                filenames.append(f'DSE-DACE{N}_workdir{i}.pt')
                # get stats to calculate mean and std of meaningful attributes
                for key in mean.keys():
                    mean[key] += graph[key].mean(dim=0)
                    mean2[key] += (graph[key]**2).mean(dim=0)

        with open(os.path.join(self.processed_dir, 'filenames.pkl'), 'wb') as file:
            pickle.dump(filenames, file)
        mean = {key: mean[key]/len(self.processed_file_names) for key in mean.keys()}
        mean2 = {key: mean2[key]/len(self.processed_file_names) for key in mean.keys()}
        std = {key: torch.sqrt(mean2[key]-mean[key]**2) for key in mean.keys()}
        torch.save(mean, os.path.join(self.processed_dir, "means.pt"))
        torch.save(std, os.path.join(self.processed_dir, "stds.pt"))

    
    def normalize_(self, graph):
        for key in self.mean.keys():
            graph[key] = (graph[key] - self.mean[key]) / self.std[key]


    def de_normalize_(self, graph):
        for key in self.mean.keys():
            graph[key] = graph[key] * self.std[key] + self.mean[key]


    def get(self, idx):
        filename = os.path.join(self.processed_dir, self.processed_file_names[idx])
        graph = torch.load(filename, weights_only=False)
        self.normalize_(graph)
        x = torch.cat([graph[key] for key in self.metadata['node_features']], dim=1)
        edge_attr = torch.cat([graph[key] for key in self.metadata['edge_features']], dim=1)
        target = torch.cat([graph[key] for key in self.metadata['target_fields']], dim=1)
        
        if self.split == 'train':
            return Data(**{'x': x, 'edge_attr': edge_attr, 'edge_index': graph.edge_index, 'target': target})
        elif self.split == 'test':
            self.de_normalize_(graph)
            return Data(**{'x': x, 'edge_attr': edge_attr, 'edge_index': graph.edge_index, 'target': target,
                        # I am pasing the potential as well to calculate the loss on the original data
                        # so that it is more immediate to compare predicting the field vs only the solenoial part
                        'velocity_x_potential': graph.velocity_x_potential,     
                        'velocity_y_potential': graph.velocity_y_potential,
                        'velocity_x': graph.velocity_x,
                        'velocity_y': graph.velocity_y,
                        'pressure': graph.pressure})    
    
    def get_to_plot(self, idx):
        filename = os.path.join(self.processed_dir, self.processed_file_names[idx])
        graph = torch.load(filename, weights_only=False)
        self.normalize_(graph)
        x = torch.cat([graph[key] for key in self.metadata['node_features']], dim=1)
        edge_attr = torch.cat([graph[key] for key in self.metadata['edge_features']], dim=1)
        target = torch.cat([graph[key] for key in self.metadata['target_fields']], dim=1)
        return Data(**{'x': x, 'edge_attr': edge_attr, 'edge_index': graph.edge_index, 'target': target, 
                       'pos': graph['pos'], 'potential_solution': graph.potential_solution, 
                       'actual_potential_diff': graph.actual_potential_diff,
                       'node_type_one_hot': graph.node_type_one_hot,
                       'body_coords': graph.body_coords})   
    
    def get_to_plot_custom_metadata(self, metadata, idx):
        filename = os.path.join(self.processed_dir, self.processed_file_names[idx])
        graph = torch.load(filename, weights_only=False)
        self.normalize_(graph)
        x = torch.cat([graph[key] for key in metadata['node_features']], dim=1)
        edge_attr = torch.cat([graph[key] for key in metadata['edge_features']], dim=1)
        target = torch.cat([graph[key] for key in metadata['target_fields']], dim=1)
        return Data(**{'x': x, 'edge_attr': edge_attr, 'edge_index': graph.edge_index, 'target': target, 
                       'pos': graph['pos'], 'potential_solution': graph.potential_solution, 
                       'actual_potential_diff': graph.actual_potential_diff,
                       'node_type_one_hot': graph.node_type_one_hot,
                       'body_coords': graph.body_coords})
    
    def get_custom_metadata(self, idx, metadata):
        filename = os.path.join(self.processed_dir, self.processed_file_names[idx])
        graph = torch.load(filename, weights_only=False)
        self.normalize_(graph)
        x = torch.cat([graph[key] for key in metadata['node_features']], dim=1)
        edge_attr = torch.cat([graph[key] for key in metadata['edge_features']], dim=1)
        y = torch.cat([graph[key] for key in metadata['target_fields']], dim=1)
        return Data(**{'x': x, 'edge_attr': edge_attr, 'edge_index': graph.edge_index, 'target': y})


    def len(self):
        return len(self.processed_file_names)

