import subprocess
import copy

import torch



def get_freest_gpu():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # List of free memory per GPU
    memory_free = [int(x) for x in result.strip().split('\n')]
    best_gpu = max(range(len(memory_free)), key=lambda i: memory_free[i])
    return f'cuda:{best_gpu}'


def get_device(metadata):
    if torch.cuda.is_available():
        if metadata['training_dynamics']['device'] == 'most_free':
            try:
                device = get_freest_gpu()
            except:
                print('fetching most free gpu failed. falling back on cuda:2')
                device = 'cuda:2'  # default on cuda:2 if the process fails
        else:
            davice = metadata['device']
    else:
        device = 'cpu'
    return device



def get_ordered_body_coords(graph):
    if hasattr(graph, 'body_coords'):
        pos_body = graph.body_coords
    else:
        pos_body = graph.pos[graph.node_type_one_hot[:,0]==1]
    
    right = pos_body[pos_body[:,0]>0]
    right = right[right[:,1].argsort(descending=True)]
    left = pos_body[pos_body[:,0]<=0]
    left = left[left[:,1].argsort(descending=False)]

    ordered_body_coords = torch.cat([right, left], dim=0)

    return ordered_body_coords



def get_original_scale(preds, metadata, dataset, potential):
    preds = copy.deepcopy(preds)
    # de-normalize
    dim = 0
    for field in metadata['target_fields']:
        size = dataset.mean[field].size(0)
        preds[:,dim:dim+size] = preds[:,dim:dim+size] * dataset.std[field] + dataset.mean[field]
        dim += size

    # add potential if needed
    dim = 0
    for field in metadata['target_fields']:
        size = dataset.mean[field].size(0)
        if field == 'actual_potential_diff':
            print('adding potential')
            potential = potential * dataset.std['potential_solution'] + dataset.mean['potential_solution']
            preds[:,dim:dim+size] += potential
        dim += size

    return preds