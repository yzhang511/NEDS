import numpy as np
import torch

class MultipleDimGauss:
    """
    Always return a zero or a one depending on the index to be unobserved
    """

    def __init__(self, data_dim=2, n_masked_vals=1, n_masks=1):
        self.data_dim = data_dim

        self.n_masked_vals = n_masked_vals
        mask_list = []
        for _ in range(n_masks):
            array_val = np.ones(data_dim)
            array_val[-n_masked_vals:] = 0
            np.random.shuffle(array_val)
            mask_list.append(array_val)

        self.indices = np.array(mask_list)
        self.n_unique_masks = self.indices.shape[0]

        assert (
            self.data_dim == self.indices.shape[1]
        ), " WARNING Dimension Mismatch: masks and data"

    def __call__(self, batch, choiceval=None):
        if choiceval is None:
            mask = torch.ones_like(batch)
            for i in range(batch.shape[0]):
                choice = np.random.choice(self.n_unique_masks)
                mask[i, :] = torch.tensor(self.indices[choice])
        elif choiceval in range(self.n_unique_masks):
            choice = choiceval
            mask = torch.tensor(self.indices[choice]).repeat(batch.shape[0], 1)
        else:
            mask = torch.ones_like(batch)
            print(
                "WARNING: provided mask choice ",
                choiceval,
                " not in valid range. Randomly sampling mask instead.",
            )
            for i in range(batch.shape[0]):
                choice = np.random.choice(self.n_unique_masks)
                mask[i, :] = torch.tensor(self.indices[choice])

        return mask