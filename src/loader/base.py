import os
import torch
import pickle
import numpy as np
from utils.dataset_utils import get_binned_spikes_from_sparse
from torch.utils.data.sampler import Sampler
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset
from numpy.random import default_rng

def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == len(seq):
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-len(seq),
                    *seq[0].shape
                )
            ) * pad_value,  
        ],
        axis=0,
    )

def _pad_seq_left_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == len(seq):
        return seq
    return np.concatenate(
        [
            np.ones(
                (
                    n-len(seq),
                    *seq[0].shape
                )
            ) * pad_value,
            seq,
        ],
        axis=0,
    )

def _wrap_pad_temporal_right_to_n(
    seq: np.ndarray,
    n: int
    ) -> np.ndarray:
    # input shape is [n_time_steps, n_neurons]
    # pad along time dimension, wrap around along space dimension
    if n == len(seq):
        return seq
    return np.pad(
        seq,
        ((0, n-seq.shape[0]), (0, 0)),
        mode='wrap'
    )
    
def _wrap_pad_neuron_up_to_n(
    seq: np.ndarray,
    n: int
    ) -> np.ndarray:
    # input shape is [n_time_steps, n_neurons]
    # pad along neuron dimension, wrap around along time dimension
    if n == len(seq[0]):
        return seq
    return np.pad(
        seq,
        ((0, 0), (0, n-seq.shape[1])),
        mode='wrap'
    )

def _attention_mask(
    seq_length: int,
    pad_length: int,
    ) -> np.ndarray:
    mask = np.ones(seq_length)
    if pad_length:
        mask[-pad_length:] = 0
    else:
        mask[:pad_length] = 0
    return mask

def _spikes_timestamps(
    seq_length: int,
    bin_size: float = 0.02,
    ) -> np.ndarray:
    return np.arange(0, seq_length * bin_size, bin_size)

def _spikes_mask(
    seq_length: int,
    mask_ratio: float = 0.1,
    ) -> np.ndarray:
    # output 0/1
    return np.random.choice([0, 1], size=(seq_length,), p=[mask_ratio, 1-mask_ratio])

def _pad_spike_seq(
    seq: np.ndarray, 
    max_length: int,
    pad_to_right: bool = True,
    pad_value: float = 0.,
) -> np.ndarray:
    pad_length = 0
    seq_len = seq.shape[0]
    if seq_len > max_length:
        seq = seq[:max_length]
    else: 
        if pad_to_right:
            pad_length = max_length - seq_len
            seq = _pad_seq_right_to_n(seq, max_length, pad_value)
        else:
            pad_length = seq_len - max_length
            seq = _pad_seq_left_to_n(seq, max_length, pad_value)
    return seq, pad_length



def get_length_grouped_indices(lengths, batch_size, shuffle=True, mega_batch_mult=None, generator=None):
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if shuffle:
        indices = torch.randperm(len(lengths), generator=generator)
    else:
        indices = torch.arange(len(lengths))
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return sum(megabatches, [])



def get_length_grouped_indices_stitched(lengths, batch_size, generator=None):
    # sort indices by length
    sorted_indices = np.argsort(lengths)
    # random indices in same length group
    group_indicies = []
    group_lengths = []
    group = []
    for i, idx in enumerate(sorted_indices):
        if i == 0:
            group.append(idx)
            group_lengths.append(lengths[idx])
        elif lengths[idx] == group_lengths[-1]:
            group.append(idx)
        else:
            group_indicies.append(group)
            group = [idx]
            group_lengths.append(lengths[idx])
    group_indicies.append(group)
    group_indicies = sum(group_indicies,[])
    # makke group_indice a multiple of batch_size
    batch_group_indicies = []
    for i in range(0, len(group_indicies), batch_size):
        batch_group_indicies.append(group_indicies[i:i+batch_size])
    if generator is not None:
        generator.shuffle(batch_group_indicies)
    else:
        np.random.shuffle(batch_group_indicies)
    batch_group_indicies = sum(batch_group_indicies, [])
    batch_group_indicies = [int(i) for i in batch_group_indicies]
    return batch_group_indicies


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        shuffle: Optional[bool] = True,
        model_input_name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_input_name = model_input_name if model_input_name is not None else "input_ids"
        if lengths is None:
            if not isinstance(dataset[0], dict) or self.model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{self.model_input_name}' key."
                )
            lengths = [len(feature[self.model_input_name]) for feature in dataset]
        self.lengths = lengths
        self.shuffle = shuffle

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.shuffle)
        return iter(indices)



class SessionSampler(Sampler):
    """Custom Sampler that batches data by session ID (eid)."""
    def __init__(self, dataset, generator, shuffle=True, seed=42):
        self.data_source = dataset
        self.shuffle = shuffle
        self.generator = generator
        self.indices_by_eid = self._group_by_eid()
        
    def _group_by_eid(self):
        from collections import defaultdict
        indices_by_eid = defaultdict(list)
        for idx, data in enumerate(self.data_source):
            indices_by_eid[data["eid"]].append(idx)
        return indices_by_eid

    def __iter__(self):
        group_indices = list(self.indices_by_eid.values())
        if self.shuffle:
            shuffled_indices = torch.randperm(len(group_indices), generator=self.generator)
            group_indices = [group_indices[ind] for ind in shuffled_indices]
        for indices in group_indices:
            if self.shuffle:
                shuffled_indices = torch.randperm(len(indices), generator=self.generator)
                indices = [indices[ind] for ind in shuffled_indices]
            yield from indices

    def __len__(self):
        return len(self.data_source)



def calculate_weights(labels):
    unique_classes = np.unique(labels)
    class_counts = np.zeros(len(unique_classes))
    for i, c in enumerate(unique_classes):
        class_counts[i] = (np.array(labels) == c).sum()
    class_weights = 1.0 / class_counts
    weights = np.array([class_weights[int(l)] for l in labels])
    return weights
    
class WeightedSessionSampler(Sampler):
    def __init__(self, dataset, shuffle=True, seed=42, replacement=True):
        self.seed = seed
        self.data_source = dataset
        self.shuffle = shuffle
        self.random_state = default_rng(seed)
        self.indices_by_eid, self.labels_by_eid, self.weights_by_eid, self.within_group_indices_by_eid = self._group_by_eid()
        self.random_indices = list(range(len(self.indices_by_eid)))
        self.replacement = replacement
        
    def _group_by_eid(self):
        from collections import defaultdict
        indices_by_eid = defaultdict(list)
        labels_by_eid = defaultdict(list)
        weights_by_eid = {}
        within_group_indices_by_eid = []
        for idx, data in enumerate(self.data_source):
            indices_by_eid[data['eid']].append(int(idx))
            labels_by_eid[data['eid']].append(data["target"][0][-1]) 
        for k, v in labels_by_eid.items():
            weights = calculate_weights(v)
            weights_by_eid[k] = weights / weights.sum()
        for k, v in indices_by_eid.items():
            within_group_indices_by_eid.append(list(range(len(v))))
        return indices_by_eid, labels_by_eid, weights_by_eid, within_group_indices_by_eid

    def __iter__(self):
        group_indices = list(self.indices_by_eid.values())
        group_labels = list(self.labels_by_eid.values())
        group_weights = list(self.weights_by_eid.values())
        within_group_indices = self.within_group_indices_by_eid.copy()
        
        if self.shuffle:
            np.random.shuffle(self.random_indices)
            group_indices = [group_indices[i] for i in self.random_indices]
            group_labels = [group_labels[i] for i in self.random_indices]
            group_weights = [group_weights[i] for i in self.random_indices]
            within_group_indices = [within_group_indices[i] for i in self.random_indices]
            for group_idx, indices in enumerate(group_indices):
                np.random.shuffle(within_group_indices[group_idx])
                indices = [indices[i] for i in within_group_indices[group_idx]]
                group_weights[group_idx] = [
                    group_weights[group_idx][i] for i in within_group_indices[group_idx]
                ]
                upsampled_indices = np.random.choice(
                    indices,
                    size=len(indices),
                    p=group_weights[group_idx],
                    replace=self.replacement
                ).tolist()
                yield from upsampled_indices
        else:
            for indices in group_indices:
                yield from indices

    def __len__(self):
        return len(self.data_source) 


class LengthStitchGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_input_name = model_input_name if model_input_name is not None else "input_ids"
        if lengths is None:
            if not isinstance(dataset[0], dict) or model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{self.model_input_name}' key."
                )
            lengths = [len(feature[self.model_input_name]) for feature in dataset]
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices_stitched(self.lengths, self.batch_size)
        return iter(indices)


def get_npy_files(data_dir, mode, eids):
    assert type(eids) == list
    # get all the npy files in the data directory
    data_dir = os.path.join(data_dir, mode)
    data_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    # only remain files with .npy extension
    data_paths = [f for f in data_paths if f.endswith('.npy')]
    # Sort the data paths first by eid then by sample index
    data_paths.sort(key=lambda x: (os.path.basename(x).split('_')[0], int(os.path.basename(x).split('_')[1].split('.')[0])))
    # filter by eid
    # the path contains one of the eids
    data_paths = [f for f in data_paths if any([eid in f for eid in eids])]
    return data_paths


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        target=None,
        pad_value = -1.,
        max_time_length = 5000,
        max_space_length = 1000,
        bin_size = 0.05,
        mask_ratio = 0.1,
        pad_to_right = True,
        sort_by_depth = False,
        sort_by_region = False,
        load_meta = False,
        brain_region = 'all',
        dataset_name = "ibl",
        stitching = False,
        data_dir = None,
        mode = "train",
        eids = None,
    ) -> None:

        if data_dir is not None:
            self.data_paths = get_npy_files(data_dir, mode, eids)
        else:
            self.data_paths = None
            self.dataset = dataset
            self.target = target
            self.pad_value = pad_value
            self.sort_by_depth = sort_by_depth
            self.sort_by_region = sort_by_region
            self.max_time_length = max_time_length
            self.max_space_length = max_space_length
            self.bin_size = bin_size
            self.pad_to_right = pad_to_right
            self.mask_ratio = mask_ratio
            self.brain_region = brain_region
            self.load_meta = load_meta
            self.dataset_name = dataset_name
            self.stitching = stitching

    def _preprocess_h5_data(self, data, idx):
        spike_data, rates, _, _ = data
        spike_data, rates = spike_data[idx], rates[idx]
        # print(spike_data.shape, rates.shape)
        spike_data, pad_length = _pad_spike_seq(spike_data, self.max_time_length, self.pad_to_right, self.pad_value)
        # add attention mask
        attention_mask = _attention_mask(self.max_time_length, pad_length).astype(np.int64)
        # add spikes timestamps
        spikes_timestamps = _spikes_timestamps(self.max_time_length, 1)
        spikes_timestamps = spikes_timestamps.astype(np.int64)

        spike_data = spike_data.astype(np.float32)
        return {"spikes_data": spike_data, 
                "rates": rates, 
                "spikes_timestamps": spikes_timestamps, 
                "attention_mask": attention_mask}

    def _preprocess_ibl_data(self, data):

        # Get sparse data and spike indices
        spikes_sparse_data = [data['spikes_sparse_data']]
        spikes_sparse_indices = [data['spikes_sparse_indices']]
        spikes_sparse_indptr = [data['spikes_sparse_indptr']]
        spikes_sparse_shape = [data['spikes_sparse_shape']]

        # Get binned spikes data from sparse representation
        binned_spikes_data = get_binned_spikes_from_sparse(
            spikes_sparse_data, spikes_sparse_indices, spikes_sparse_indptr, spikes_sparse_shape
        )[0]

        # Prepare target behavior
        if self.target:
            target_behavior, target_behavior_dict = self._prepare_target_behavior(data)
        else:
            target_behavior = np.array([np.nan])

        # Prepare choice, block, and reward data
        static_vars = ['choice', 'block', 'reward']
        choice, block, reward = map(self._prepare_column_data, static_vars, [data] * len(static_vars))

        # Prepare lookup dictionaries
        choice_lookup = {'-1.0': 0, '1.0': 1}
        block_lookup = {'0.2': 0, '0.5': 1, '0.8': 2}

        # Create lookup arrays for choice and block
        _choice, _block = self._apply_lookups(choice, block, choice_lookup, block_lookup, target_behavior.shape[0])
        choice, block = np.float32(_choice[0]), np.float32(_block[0])
    
        # Combine target_behavior with choice and block
        target_behavior = np.concatenate([target_behavior, _choice, _block], axis=1).astype(np.float32)

        # Adjust neuron IDs
        include_neuron_ids = np.arange(binned_spikes_data.shape[-1]).astype(np.int64)
        binned_spikes_data = binned_spikes_data[:, include_neuron_ids].squeeze()

        # Process metadata if `load_meta` is set
        neuron_depths, neuron_regions = self._load_neuron_metadata(
            data, include_neuron_ids
        ) if self.load_meta else (np.array([np.nan]), np.array([np.nan]))

        # Sort data if specified
        binned_spikes_data, neuron_depths, neuron_regions = self._sort_data_by_depth_or_region(
            binned_spikes_data, neuron_depths, neuron_regions
        )
            
        # Pad along time and space dimensions
        binned_spikes_data, pad_time_length = self._pad_data(binned_spikes_data, self.max_time_length, axis=0)
        binned_spikes_data, pad_space_length = self._pad_data(binned_spikes_data, self.max_space_length, axis=1)

        # Prepare the attention masks
        time_attn_mask = _attention_mask(self.max_time_length, pad_time_length).astype(np.int64)
        space_attn_mask = _attention_mask(self.max_space_length, pad_space_length).astype(np.int64)

        # Generate spike timestamps and spacestamps
        spikes_timestamps = np.arange(self.max_time_length).astype(np.int64)
        spikes_spacestamps = np.arange(self.max_space_length).astype(np.int64)

        # Pad neuron_depths and neuron_regions to max_space_length
        neuron_depths = np.pad(
            neuron_depths, 
            (0, max(0, self.max_space_length - neuron_depths.shape[0])),
            constant_values=np.nan
        )

        neuron_regions = np.pad(
            neuron_regions,
            (0, max(0, self.max_space_length - neuron_regions.shape[0])),
            constant_values=""
        )

        return {
            "spikes_data": binned_spikes_data.astype(np.float32),
            "time_attn_mask": time_attn_mask,
            "space_attn_mask": space_attn_mask,
            "spikes_timestamps": spikes_timestamps,
            "spikes_spacestamps": spikes_spacestamps,
            "target": target_behavior,
            "neuron_depths": neuron_depths,
            "neuron_regions": list(neuron_regions),
            "eid": data['eid'],
            "choice": choice,
            "block": block,
            "reward": reward,
            **target_behavior_dict,
        }
    
    def _prepare_target_behavior(self, data):
        target_behavior = []
        target_behavior_dict = {}
        for beh_name in self.target:
            beh = np.array(data[beh_name], dtype=np.float32)
            target_behavior.append(beh)
            target_behavior_dict[beh_name.split('-')[0]] = beh
        return np.array(target_behavior).T, target_behavior_dict

    def _prepare_column_data(self, col_name, data):
        return np.array(data[col_name], dtype=np.float32)

    def _apply_lookups(self, choice, block, choice_lookup, block_lookup, target_len):
        _choice = np.array([choice_lookup[str(x)] for x in choice] * target_len).reshape(-1, 1)
        _block = np.array([block_lookup[str(x)] for x in block] * target_len).reshape(-1, 1)
        return _choice, _block

    def _load_neuron_metadata(self, data, include_neuron_ids):
        neuron_depths = np.array(data['cluster_depths'], dtype=np.float32)[include_neuron_ids].squeeze()
        neuron_regions = np.array(data['cluster_regions'], dtype='str')[include_neuron_ids].squeeze()
        return neuron_depths, neuron_regions

    def _sort_data_by_depth_or_region(self, binned_spikes_data, neuron_depths, neuron_regions):
        if self.sort_by_depth:
            sorted_idxs = np.argsort(neuron_depths)
        elif self.sort_by_region:
            sorted_idxs = np.argsort(neuron_regions)
        else:
            sorted_idxs = np.arange(len(neuron_depths))

        binned_spikes_data = binned_spikes_data[:, sorted_idxs]
        neuron_depths = neuron_depths[sorted_idxs]
        neuron_regions = neuron_regions[sorted_idxs]

        return binned_spikes_data, neuron_depths, neuron_regions

    def _pad_data(self, data, max_len, axis):
        data_len = data.shape[axis]
        pad_len = max_len - data_len
        if pad_len > 0:
            if axis == 0:
                data = _pad_seq_right_to_n(data, max_len, self.pad_value)
            else:
                data = _pad_seq_right_to_n(data.T, max_len, self.pad_value).T
        return data, pad_len
    
    def __len__(self):
        if self.data_paths is not None:
            return len(self.data_paths)
        elif "ibl" in self.dataset_name:
            return len(self.dataset)
        else:
            # get the length of the first tuple in the dataset
            return len(self.dataset)
        
    def __getitem__(self, idx):
        if self.data_paths is not None:
            data = np.load(self.data_paths[idx], allow_pickle=True).item()
            return data
        elif "ibl" in self.dataset_name:
            return self._preprocess_ibl_data(self.dataset[idx])
        else:
            return self._preprocess_h5_data(self.dataset, idx)  
 