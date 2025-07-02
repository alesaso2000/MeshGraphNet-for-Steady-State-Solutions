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
    return torch.device(device)



# def get_ordered_body_coords(graph):
#     if hasattr(graph, 'body_coords'):
#         pos_body = graph.body_coords
#     else:
#         pos_body = graph.pos[graph.node_type_one_hot[:,0]==1]
    
#     right = pos_body[pos_body[:,0]>0]
#     right = right[right[:,1].argsort(descending=True)]
#     left = pos_body[pos_body[:,0]<=0]
#     left = left[left[:,1].argsort(descending=False)]

#     ordered_body_coords = torch.cat([right, left], dim=0)

#     return ordered_body_coords
 

def get_ordered_body_coords(graph):
    if hasattr(graph, 'body_coords'):
        pos_body = graph.body_coords
    else:
        pos_body = graph.pos[graph.node_type_one_hot[:,0]==1]
    
    center = pos_body.mean(dim=0)
    centered = pos_body - center
    angles = torch.atan2(centered[:, 1], centered[:, 0])
    sorted_idx = torch.argsort(angles)
    sorted_pos_body = pos_body[sorted_idx]
    return sorted_pos_body



def get_base_field_name(field):
    """
    I call the fields something like 'velocity_x_solenoidal and want back only velocity_x
    """
    return '_'.join(field.split('_')[:2])


def get_original_scale(preds, metadata, means, stds, graph):
    """
    Returns a clone of the prediction on the same scale as the original data
    """
    preds = torch.clone(preds)
    for dim, field in enumerate(metadata['target_fields']):
        preds[:,dim] = preds[:,dim] * stds[field] + means[field]
        if 'solenoidal' in field:
            preds[:,dim] += graph[get_base_field_name(field)+'_potential'].squeeze()
    return preds