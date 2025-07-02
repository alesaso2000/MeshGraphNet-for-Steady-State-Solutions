import re, os
from typing import List

import torch
from torch_geometric.data import Data
import numpy as np

# from utils_potential import body_panels, sol_pot, mk_coeff_xy
# from utils import order_body_coords


############ Functions to calculate potential ################
import numpy as np


def body_panels(xy):
    xpan = np.concatenate([xy[0:-1, 0:1], xy[1:, 0:1]], axis=1)
    ypan = np.concatenate([xy[0:-1, 1:2], xy[1:, 1:2]], axis=1)
    dx = np.diff(xpan, axis=1)
    dy = np.diff(ypan, axis=1)
    thpan = np.arctan2(dy, dx)
    xycpan = 0.5*np.concatenate([np.sum(xpan, axis=1, keepdims=True),
                                  np.sum(ypan, axis=1, keepdims=True)], axis=1)
    nor = np.concatenate([-np.sin(thpan), np.cos(thpan)], axis=1)
    return xycpan, xpan, ypan, thpan, nor



def mk_coeff_local_body(xycpan, thpan, xpan, ypan):
    npan = len(thpan)
    Anor = np.zeros((npan, npan))
    Atan = np.zeros((npan, npan))
    for ip in range(npan):
        ising = np.zeros((npan, 1))
        ising[ip] = 1
        xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
        thp = thpan[ip]
        AA = source_coeff_vel_local(xycpan, thpan, xyp, thp, ising)
        Anor[:, ip:ip+1] = AA[:, 1:2]
        Atan[:, ip:ip+1] = AA[:, 0:1]
    return Anor, Atan


# def mk_coeff_xy_body(xy, thpan, xpan, ypan):
#     npan = len(thpan)
#     Ax = np.zeros((npan, npan))
#     Ay = np.zeros((npan, npan))
#     for ip in range(npan-1):
#         ising = np.zeros((npan, 1))
#         ising[ip] = 1
#         xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
#         thp = thpan[ip]
#         AA = source_coeff_vel_bodyfixed(xy, xyp, thp, ising)
#         Ay[:, ip:ip + 1] = AA[:, 1:2]
#         Ax[:, ip:ip + 1] = AA[:, 0:1]
#     return Ax, Ay


def mk_coeff_xy(xy, thpan, xpan, ypan):
    npan = len(thpan)
    Ax = np.zeros((len(xy), npan))
    Ay = np.zeros((len(xy), npan))
    ising = np.zeros((len(xy), 1))
    for ip in range(npan-1):
        xyp = np.concatenate([xpan[ip, 0:2].reshape(-1, 1), ypan[ip, 0:2].reshape(-1, 1)], axis=1)
        thp = thpan[ip]
        AA = source_coeff_vel_bodyfixed(xy, xyp, thp, ising)
        Ay[:, ip:ip + 1] = AA[:, 1:2]
        Ax[:, ip:ip + 1] = AA[:, 0:1]
    return Ax, Ay




def source_coeff_vel_local(xycpan, thpan, xyp, thp, ising):

    lc1 = xycpan - xyp[0, 0:2]
    lc2 = xycpan - xyp[1, 0:2]

    rij = np.sqrt(np.sum(lc1**2, axis=1, keepdims=True))
    rijp1 = np.sqrt(np.sum(lc2**2, axis=1, keepdims=True))

    colp_sin = np.sin(thpan)
    colp_cos = np.cos(thpan)
    p_sin = np.sin(thp)
    p_cos = np.cos(thp)

    betaij = np.arctan2(lc2[:, 1:2]*lc1[:, 0:1] - lc2[:, 0:1]*lc1[:, 1:2],
                           lc2[:, 0:1]*lc1[:, 0:1] + lc2[:, 1:2]*lc1[:, 1:2])

    lr2r1 = np.log(rijp1/rij)

    inv2pi = 1/2/np.pi

    A0 = inv2pi * (
        (
            betaij * (colp_sin*p_cos -
                      colp_cos*p_sin) -
            lr2r1 * (colp_cos*p_cos +
                     colp_sin*p_sin)
        ) * (1 - ising) +
        (
            np.pi * (colp_sin*p_cos -
                        colp_cos*p_sin)
        ) * ising
    )

    A1 = inv2pi * (
            (
                    betaij * (colp_cos * p_cos +
                              colp_sin * p_sin) +
                    lr2r1 * (colp_sin * p_cos -
                             colp_cos * p_sin)
            ) * (1 - ising) +
            (
                    np.pi * (colp_cos * p_cos +
                                colp_sin * p_sin)
            ) * ising
    )

    AA = np.concatenate([A0, A1], axis=1)
    return AA


def source_coeff_vel_bodyfixed(xycpan, xyp, thp, ising):

    lc1 = xycpan - xyp[0, 0:2]
    lc2 = xycpan - xyp[1, 0:2]

    rij = np.sqrt(np.sum(lc1**2, axis=1, keepdims=True))
    rijp1 = np.sqrt(np.sum(lc2**2, axis=1, keepdims=True))

    p_sin = np.sin(thp)
    p_cos = np.cos(thp)

    betaij = np.arctan2(lc2[:, 1:2]*lc1[:, 0:1] - lc2[:, 0:1]*lc1[:, 1:2],
                           lc2[:, 0:1]*lc1[:, 0:1] + lc2[:, 1:2]*lc1[:, 1:2])

    '''
    lr2r1 = torch.tensor([-1, 1]).to(params['device'])
    for i in range(len(rij)):
        if (rij[i] > 0) & (rijp1[i] > 0):
            lr2r1 = torch.cat([lr2r1, torch.log(rijp1[i] / rij[i])], dim=0)
        else:
            lr2r1 = torch.cat([lr2r1, torch.tensor([0]).to(params['device'])], dim=0)
    '''

    lr2r1 = np.log(rijp1/rij)

    inv2pi = 1/2/np.pi

    A0 = inv2pi * (
        (
            betaij * (- p_sin) -
            lr2r1 * (+ p_cos)
        ) * (1 - ising) +
        (
            np.pi * (- p_sin)
        ) * ising
    )

    A1 = inv2pi * (
            (
                    betaij * (+ p_cos) +
                    lr2r1 * (- p_sin)
            ) * (1 - ising) +
            (
                    np.pi * (+ p_cos)
            ) * ising
    )

    AA = np.concatenate([A0, A1], axis=1)
    return AA


def sol_pot(xy, xpan, ypan, thpan, norpan):
    An, _ = mk_coeff_local_body(xy, thpan, xpan, ypan)
    B = norpan[:, 0:1]
    sigmax = np.linalg.solve(An, B)
    B = norpan[:, 1:2]
    sigmay = np.linalg.solve(An, B)
    return sigmax, sigmay
#####################################################################################################################


#################### Extract graph ##################################################


# def order_body_coords(pos_body):
#     right = pos_body[pos_body[:,0]>0]
#     right = right[right[:,1].argsort(descending=True)]
#     left = pos_body[pos_body[:,0]<=0]
#     left = left[left[:,1].argsort(descending=False)]

#     ordered_body_coords = torch.cat([right, left], dim=0)

#     return ordered_body_coords


def order_body_coords(pos_body):
    center = pos_body.mean(dim=0)
    centered = pos_body - center
    angles = torch.atan2(centered[:, 1], centered[:, 0])
    sorted_idx = torch.argsort(angles)
    return pos_body[sorted_idx]


def extract_graph(file_path, node_features, edge_features=['edge']):
    """
    Extracts the data from the OPENFoam output and makes it into a graph.
    Since the simulation is finite volume, I will construct and use the graph which connects the cell centers (of adjecent cells),
    not the mesh that has as edges the edge of the cell 
    """

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

            return node_type_one_hot, nodes_of_each_type


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
        node_type_one_hot, nodes_of_each_type = extract_node_type(owners)
        points = extract_points()
        faces = extract_faces()

        return owners, neighbours, points, faces, node_type_one_hot, nodes_of_each_type


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

        # # REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # def extract_pressure(which: str, n_line=19, start_line=21):
        #     with open(os.path.join(file_path, which), 'r') as f:
        #         lines = f.read().splitlines()
        #     n = int(lines[n_line])
        #     return torch.tensor([float(l) for l in lines[start_line: start_line+n]]).unsqueeze(-1)
        # ##########################################
        
        def extract_body_coords():
            with open(os.path.join(file_path, 'C'), 'r') as f:
                lines = f.read().splitlines()
            prof_extruded_line = lines.index('    prof_extruded')
            n = int(lines[prof_extruded_line + 4])
            return torch.tensor([[float(c) for c in line[1:-1].split()[:2]] for line in lines[prof_extruded_line+6 : prof_extruded_line+6+n]])

        centers = extract('C')
        # velocity = extract('UMean')
        # pressure = extract_pressure('pMean')
        body_coords = extract_body_coords()

        return centers, body_coords #, velocity, pressure


    # def extract_bspline_points(file_path):
    #     with open(file_path) as f:
    #         line = f.read().split()
    #     points = [float(p) for p in line]
    #     return torch.tensor(points) # west_x, east_x, north_y, south_y

    
    # def check_orderability(body_coords, bspline_points):
    #     """"
    #     Checks that the nodes on the object are easily orderable (i.e. the tips are not higher or lower than the center)
    #     """
    #     check = body_coords[:,1].min() >= bspline_points[-1] and body_coords[:,1].max() <= bspline_points[-2]
    #     return check
    
    def get_potential_solution(body_coords, pos):
        ordered_body_coords = order_body_coords(body_coords)
        ordered_body_coords = torch.cat([ordered_body_coords, ordered_body_coords[0].unsqueeze(0)], dim=0)
        xycpan, xpan, ypan, thpan, nor = body_panels(ordered_body_coords.numpy())
        sigma_x, sigma_y = sol_pot(xycpan, xpan, ypan, thpan, nor)
        Ax_field, Ay_field = mk_coeff_xy(pos.numpy(), thpan, xpan, ypan)  
        u_pot_field = np.matmul(Ax_field, sigma_x) + np.matmul(Ax_field, 0*sigma_y) + 1
        u_pot_field = torch.from_numpy(u_pot_field).float()
        v_pot_field = np.matmul(Ay_field, sigma_x) + np.matmul(Ay_field, 0*sigma_y) + 0
        v_pot_field = torch.from_numpy(v_pot_field).float()
        potential_solution = torch.cat([u_pot_field, v_pot_field], dim=1)
        return potential_solution
    

    def get_fourier_feature(pos):
        frequencies = torch.tensor([2, 4, 8, 16, 32]) * torch.pi  # shape (5,)
        minmaxed_positions = (pos - pos.min(dim=0).values) / (pos.max(dim=0).values - pos.min(dim=0).values) # shape (N,2)
        arg = minmaxed_positions.unsqueeze(2) * frequencies  # shape(N,2,4)
        sines = torch.sin(arg).reshape(pos.shape[0], pos.shape[1]*frequencies.shape[0])  # (points, coordinate, freq) -> (points, sin([x(freq[0]), y(freq[0]),..., x(freq[4]), y(freq[4])]))
        cosines = torch.cos(arg).reshape(pos.shape[0], pos.shape[1]*frequencies.shape[0]) 
        return torch.cat([sines, cosines], dim=1)


    owners, neighbours, _, _, node_type_one_hot, nodes_of_each_type = extract_polymesh(os.path.join(file_path, 'constant/polyMesh'))
    centers, body_coords = extract_fields(os.path.join(file_path, '0'))
    # bspline_points = extract_bspline_points(os.path.join(file_path, f'input.txt'))
    # if not check_orderability(body_coords, bspline_points):
    #     raise ValueError('The points are not orderable')
    edge_index, distance_vector, distance_magnitude = make_edges(owners, neighbours, centers)
    distance_from_obj_vec, distance_from_obj_mag = get_distance_from_object(node_type_one_hot, centers)
    potential_solution = get_potential_solution(body_coords, centers)
    fourier_feature = get_fourier_feature(centers)

    # Normalize and package into dict
    means = torch.load('/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/train/processed/means.pt', weights_only=False)
    stds = torch.load('/davinci-1/work/dsalvatore/CFD/meshgraphnet_cubotto/data/train/processed/stds.pt', weights_only=False)
    features = {}
    features['distance_magnitude'] = (distance_magnitude - means['distance_magnitude']) / stds['distance_magnitude']
    features['distance_vector'] = (distance_vector - means['distance_vector']) / stds['distance_vector']
    features['distance_from_obj_mag'] = (distance_from_obj_mag - means['distance_from_obj_mag']) / stds['distance_from_obj_mag']
    features['distance_from_obj_vec'] = (distance_from_obj_vec - means['distance_from_obj_vec']) / stds['distance_from_obj_vec']
    features['velocity_x_potential'] = (potential_solution[:,0].unsqueeze(1) - means['velocity_x_potential']) / stds['velocity_x_potential']
    features['velocity_y_potential'] = (potential_solution[:,1].unsqueeze(1) - means['velocity_y_potential']) / stds['velocity_y_potential']
    features['pos'] = (centers - means['pos']) / stds['pos']
    features['fourier_feature'] = (fourier_feature - means['fourier_feature']) / stds['fourier_feature']
    # Package without normalizing
    features['node_type_one_hot'] = node_type_one_hot
    
    return Data(**{
        'x': torch.cat([features[key] for key in node_features], dim=-1),
        'edge_attr' : torch.cat([features[key] for key in edge_features], dim=-1),
        'edge_index': edge_index,
        'nodes_of_each_type': nodes_of_each_type,
        'body_coords': body_coords,
        'pos': centers,
        'velocity_x_potential': potential_solution[:,0].unsqueeze(1),
        'velocity_y_potential': potential_solution[:,1].unsqueeze(1),
        # 'velocity_x': velocity[:,0].unsqueeze(1),
        # 'velocity_y': velocity[:,1].unsqueeze(1),
        # 'pressure': pressure
    })




def get_correct_velocity_pressure(file_path='./1000'):
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

    velocity = extract('UMean')
    pressure = extract_pressure('pMean')
    
    return velocity, pressure
