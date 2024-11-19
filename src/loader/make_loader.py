import numpy as np
import torch
from loader.base import (
    BaseDataset, 
    LengthStitchGroupedSampler, 
    LengthGroupedSampler, 
    SessionSampler,
    WeightedSessionSampler
)
from torch.utils.data.sampler import WeightedRandomSampler

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_weights(labels):
    unique_classes = np.unique(labels)
    class_counts = np.zeros(len(unique_classes))
    for i, c in enumerate(unique_classes):
        class_counts[i] = (np.array(labels) == c).sum()
    class_weights = 1.0 / class_counts
    weights = np.array([class_weights[int(l)] for l in labels])
    return weights

def make_loader(
    dataset, 
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
    shuffle = True,
    weighted_sampler=False,
):
    
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

    generator = torch.Generator()
    generator.manual_seed(seed)

    if stitching:
        if weighted_sampler:
            assert weighted_sampler == False, "Weighted sampler not seeded yet."
            sampler = WeightedSessionSampler(dataset=dataset, shuffle=shuffle, seed=seed) 
        else:
            sampler = SessionSampler(dataset=dataset, generator=generator, shuffle=shuffle, seed=seed)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler, 
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=True,
        )
    else:
        if weighted_sampler:
            labels = [x["target"][0][-1] for x in dataset]   # weight samples according to block
            weights = torch.from_numpy(calculate_weights(labels)).double()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), generator=generator)
            dataloader = torch.utils.data.DataLoader(
                dataset, sampler=sampler, batch_size=batch_size, 
                worker_init_fn=seed_worker, generator=generator, pin_memory=True,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, 
                worker_init_fn=seed_worker, generator=generator, pin_memory=True,
            )

    return dataloader
