import numpy as np
import torch
from loader.base import (
    BaseDataset, 
    LengthStitchGroupedSampler, 
    LengthGroupedSampler, 
    SessionSampler,
    WeightedSessionSampler
)
#from torch.utils.data.sampler import WeightedRandomSampler

def calculate_weights(labels):
    unique_classes = np.unique(labels)
    class_counts = np.zeros(len(unique_classes))
    for i, c in enumerate(unique_classes):
        class_counts[i] = (np.array(labels) == c).sum()
    class_weights = 1.0 / class_counts
    weights = np.array([class_weights[int(l)] for l in labels])
    return weights

def make_loader(dataset, 
                 batch_size, 
                 target = None,
                 pad_to_right = True,
                 sort_by_depth = False,
                 sort_by_region = False,
                 pad_value = 0.,
                 max_time_length = 5000,
                 max_space_length = 100,
                 bin_size = 0.05,
                 brain_region = 'all',
                 load_meta=False,
                 use_nemo=False,
                 dataset_name = "ibl",
                 stitching = False,
                 seed=42,
                 shuffle = True):
    
    dataset = BaseDataset(dataset=dataset, 
                          target=target,
                          pad_value=pad_value,
                          max_time_length=max_time_length,
                          max_space_length=max_space_length,
                          bin_size=bin_size,
                          pad_to_right=pad_to_right,
                          dataset_name=dataset_name,
                          sort_by_depth = sort_by_depth,
                          sort_by_region = sort_by_region,
                          brain_region = brain_region,
                          load_meta=load_meta,
                          use_nemo=use_nemo,
                          stitching=stitching
            )
    
    print(f"len(dataset): {len(dataset)}")

    if stitching:
        #####
        # session_sampler = SessionSampler(dataset=dataset, shuffle=shuffle, seed=seed)
        #labels = [x["target"][0][-1] for x in dataset] # block variable
        #weights = torch.from_numpy(calculate_weights(labels)).double()
        #weighted_sampler = WeightedRandomSampler(weights, num_samples=len(weights))

        weighted_sampler = WeightedSessionSampler(dataset=dataset, shuffle=shuffle, seed=seed) 
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            # sampler=session_sampler, 
            sampler=weighted_sampler,
            batch_size=batch_size,
        )
        #####
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
