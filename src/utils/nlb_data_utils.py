import numpy as np
import pickle
import yaml
import copy
import os
from sklearn.model_selection import train_test_split

################################################################################

def load_nlb_dataset(dataset_path, bin_size_ms=5, tau_prime=13):

    Train, Val, Test = load_data(dataset_path, test_frac=0.2)
    Train, Val, Test = restrict_data(Train, Val, Test, var_group="vel")

    Delta = bin_size_ms

    for Data in [Train, Val, Test]:

        S = Data['spikes']
        Z = Data['behavior']

        # Bin spikes.
        S = [bin_spikes(sp, Delta) for sp in S]

        # Reformat observations to include recent history.
        X = [append_history(s, tau_prime) for s in S]

        # Downsample kinematics to bin width.
        Z = [z[:,Delta-1::Delta] for z in Z]

        # Remove samples on each trial for which sufficient spiking history doesn't exist.
        X = [x[:,tau_prime:,:] for x in X]
        Z = [z[:,tau_prime:] for z in Z]

        # Concatenate X and Z across trials (in time bin dimension) and rearrange dimensions.
        X = np.moveaxis(np.concatenate(X,axis=1), [0, 1, 2], [1, 0, 2])
        Z = np.concatenate(Z, axis=1).T

        # Z-score inputs.
        # X_mu = np.mean(X, axis=0)
        # X_sigma = np.std(X, axis=0)
        # X = (X - X_mu) / X_sigma
        # X_mu = X_mu
        # X_sigma = X_sigma

        # Zero-center outputs.
        Z_mu = np.mean(Z, axis=0)
        Z = Z - Z_mu
        Z_mu = Z_mu

        Data['spike'] = np.moveaxis(X, [0, 1, 2], [0, 2, 1])
        Data['finger_x_vel'] = Z[...,0]
        Data['finger_y_vel'] = Z[...,1]
        # Data['X_mu'] = X_mu
        # Data['X_sigma'] = X_sigma
        Data['finger_x_vel_mean'] = Z_mu[0]
        Data['finger_y_vel_mean'] = Z_mu[1]

        Data.pop('spikes')
        Data.pop('behavior')
        Data.pop('condition')

    meta_data = {
        "num_sessions": 1,
        "eids": {"nlb-rtt"},
        "eid_list": {"nlb-rtt": Train['spike'].shape[2]},
    }

    return Train, Val, Test, meta_data

################################################################################

def partition(condition, test_frac):

    """
    Partition trials into training and testing sets.

    Inputs
    ------
    condition: list (1 x number of trials) of condition IDs
        If there is no condition structure, all entries will be NaNs.

    test_frac: fraction of trials in the data set that should be reserved for testing

    Outputs
    -------
    train_idx: 1D numpy array of indices for trials that should be used for training

    test_idx: 1D numpy array of indices for trials that should be used for testing
   
    """

    # Number the trials.
    trial_idx = np.arange(len(condition))
    
    # Partition the data differently depending on whether there is condition structure.
    if not np.all(np.isnan(condition)):
        # Try to maintain equal representation from different conditions in the train
        # and test sets. If the test set is so small that it can't sample at least
        # one from each condition, then don't bother trying to stratify the split.
        n_conds = len(np.unique(np.array(condition)))
        n_test = int(test_frac*len(condition))
        if n_test >= n_conds:
            train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, stratify=condition, random_state=42)
        else:
            train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)
    else:
        # Divide the trials up randomly into train and test sets.
        train_idx, test_idx = train_test_split(trial_idx, test_size=test_frac, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42)

    return train_idx, val_idx, test_idx

################################################################################

def bin_spikes(spikes, bin_size):

    """
    Bin spikes in time.

    Inputs
    ------
    spikes: numpy array of spikes (neurons x time)

    bin_size: number of time points to pool into a time bin

    Outputs
    -------
    S: numpy array of spike counts (neurons x bins)
   
    """

    # Get some useful constants.
    [N, n_time_samples] = spikes.shape
    K = int(n_time_samples/bin_size) # number of time bins

    # Count spikes in bins.
    S = np.empty([N, K])
    for k in range(K):
        S[:, k] = np.sum(spikes[:, k*bin_size:(k+1)*bin_size], axis=1)

    return S

################################################################################

def bin_kin(Z, bin_size):

    """
    Bin behavioral variables in time.

    Inputs
    ------
    Z: numpy array of behavioral variables (behaviors x time)

    bin_size: number of time points to pool into a time bin

    Outputs
    -------
    Z_bin: numpy array of binned behavioral variables (behaviors x bins)
   
    """

    # Get some useful constants.
    [M, n_time_samples] = Z.shape
    K = int(n_time_samples/bin_size) # number of time bins

    # Average kinematics within bins.
    Z_bin = np.empty([M, K])
    for k in range(K):
        Z_bin[:, k] = np.mean(Z[:, k*bin_size:(k+1)*bin_size], axis=1)

    return Z_bin

################################################################################

def append_history(S, tau_prime):

    """
    Augment spike count array with additional dimension for recent spiking history.

    Inputs
    ------
    S: numpy array of spike counts (neurons x bins)

    tau_prime: number of historical time bins to add (not including current bin)

    Outputs
    -------
    S_aug: tensor of spike counts (neurons x bins x recent bins)
   
    """

    # Get some useful constants.
    [N, K] = S.shape # [number of neurons, number of bins]

    # Augment matrix with recent history.
    S_aug = np.empty([N, K, tau_prime+1])
    for i in range(-tau_prime,0):
        S_aug[:, :, i+tau_prime] = np.hstack((np.full([N,-i], np.nan), S[:, :i]))
    S_aug[:, :, tau_prime] = S

    return S_aug

################################################################################

def array2list(array, sizes, axis):

    """
    Break up a numpy array along a particular axis into a list of arrays.

    Inputs
    ------
    array: numpy array to break up

    sizes: vector of sizes for the resulting arrays along the specified axis (should sum to input array size along this axis)

    axis: axis to break up array along

    Outputs
    -------
    list_of_arrays: list where each element is a numpy array
   
    """
    
    # Get indices indicating where to divide array.
    split_idx = np.cumsum(sizes)
    split_idx = split_idx[:-1]
    
    # Split up array.
    array_of_arrays = np.split(array, split_idx, axis=axis)
    
    # Convert outer array to list.
    list_of_arrays = list(array_of_arrays)

    return list_of_arrays

################################################################################

def zero_order_hold(binned_data, bin_size):

    """
    Upsample data with zero-order hold.

    Inputs
    ------
    binned_data: numpy array (variables x bins)

    bin_size: number of samples that were pooled into each time bin

    Outputs
    -------
    unbinned_data: numpy array of zero-order-held data (variables x time)
   
    """
    
    # Preallocate unbinned data.
    n_vars, n_bins = binned_data.shape
    unbinned_data = np.zeros((n_vars, n_bins*bin_size))
    
    # Upsample data with zero-order hold.
    for k in range(n_bins):
        unbinned_data[:,k*bin_size:(k+1)*bin_size] = np.tile(np.reshape(binned_data[:,k],(-1,1)),bin_size)

    return unbinned_data

################################################################################

def pad_to_length(data, T):

    """
    Pad data with final value as needed to reach a specified length.

    Inputs
    ------
    data: numpy array (variables x time)

    T: number of desired time samples

    Outputs
    -------
    padded_data: numpy array (variables x T)
   
    """

    # Initialized padded_data as data.
    padded_data = data

    # If padding is necessary...
    n_samples = data.shape[1]
    if n_samples < T:
        final_value = data[:,-1]
        pad_len = T - n_samples
        pad = np.tile(final_value,(pad_len,1)).T
        padded_data = np.hstack((padded_data,pad))

    return padded_data

################################################################################

def compute_R2(Z, Z_hat, skip_samples=0, eval_bin_size=1):

    """
    Compute the coefficients of determination (R2).

    Inputs
    ------
    Z: list of ground truth matrices (behaviors x observations)

    Z_hat: list of predicted values matrices (behaviors x observations)

    skip_samples: number of observations to exclude at the start of each trial
        Methods that generate predictions causally may not have predictions
        for the first few bins of a trial. This input allows these bins to
        be excluded from the R2 computation.

    eval_bin_size: number of observations to bin in time (by averaging) before computing R2

    Outputs
    -------
    R2: numpy array of R2s (one per behavioral variable)

    """

    # Remove some samples at the beginning of each
    # trial that were flagged to be skipped.
    Z = [z[:,skip_samples:] for z in Z]
    Z_hat = [z[:,skip_samples:] for z in Z_hat]

    # Bin kinematics in time.
    Z = [bin_kin(z, eval_bin_size) for z in Z]
    Z_hat = [bin_kin(z, eval_bin_size) for z in Z_hat]

    # Concatenate lists.
    Z = np.concatenate(Z,1)
    Z_hat = np.concatenate(Z_hat,1)

    # Compute residual sum of squares.
    SS_res = np.sum((Z - Z_hat)**2, axis=1)

    # Compute total sum of squares.
    Z_mu = np.transpose([np.mean(Z, axis=1)] * Z.shape[1])
    SS_tot = np.sum((Z - Z_mu)**2, axis=1)

    # Compute coefficient of determination.
    R2 = 1 - SS_res/SS_tot

    return R2

################################################################################

def load_data(dataset_path, test_frac):

    """
    Load data set and split into training and testing sets.

    Inputs
    ------
    dataset: string containing filename of dataset (without .pickle extension)

    test_frac: fraction of trials in the data set that should be reserved for testing

    Outputs
    -------
    Train: dictionary containing trialized neural and behavioral data in training set

    Test: dictionary containing trialized neural and behavioral data in testing set
   
    """

    # Load data file.
    with open(dataset_path, 'rb') as f:
        Data = pickle.load(f)

    # Format condition.
    if 'condition' in Data.keys():
        Data['condition'] = list(Data['condition']) # convert from np array to list
    else:
        first_key = list(Data.keys())[0]
        n_trials = len(Data[first_key])
        Data['condition'] = [np.nan]*n_trials # if no condition key exists, make one with all NaNs

    # Partition into training and testing sets.
    train_idx, val_idx, test_idx = partition(Data['condition'], test_frac)
    Train = dict()
    Val = dict()
    Test = dict()
    for k in Data.keys():
        Train[k] = [Data[k][i] for i in train_idx]
        Val[k] = [Data[k][i] for i in val_idx]
        Test[k] = [Data[k][i] for i in test_idx]
    
    return Train, Val, Test

################################################################################

def save_data(Results, run_name):

    """
    Save decoding results.

    Inputs
    ------
    Results: dictionary containing decoding results

    run_name: filename to use for saving results (without .pickle extension)
   
    """

    # If the 'results' directory doesn't exist, create it.
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save Results as .pickle file.
    with open('results/' + run_name + '.pickle','wb') as f:
        pickle.dump(Results,f)
    
################################################################################

def restrict_data(Train, Val, Test, var_group):

    """
    Restrict training and testing data to only include a particular behavioral variable group(s).

    Inputs
    ------
    Train: dictionary containing trialized neural and behavioral data in training set

    Test: dictionary containing trialized neural and behavioral data in testing set

    var_group: string (or list of strings) containing behavioral variable group(s) to restrict to

    Outputs
    -------
    Train_b: copy of Train where behavioral data has been restricted as specified by var_group

    Test_b: copy of Test where behavioral data has been restricted as specified by var_group
   
    """
    
    # Initialize outputs.
    Train_b = dict()
    Val_b = dict()
    Test_b = dict()
    
    # Copy spikes into new dictionaries.
    Train_b['spikes'] = copy.deepcopy(Train['spikes'])
    Val_b['spikes'] = copy.deepcopy(Val['spikes'])
    Test_b['spikes'] = copy.deepcopy(Test['spikes'])
    
    # Copy condition labels (if task had condition structure).
    if 'condition' in Train:
        Train_b['condition'] = copy.deepcopy(Train['condition'])
        Val_b['condition'] = copy.deepcopy(Val['condition'])
        Test_b['condition'] = copy.deepcopy(Test['condition'])
    else:
        Train_b['condition'] = []
        Val_b['condition'] = []
        Test_b['condition'] = []
        
    # Copy over relevant behavioral variables (and concatenate them into a single variable).
    if isinstance(var_group,str): # if var_group is just one variable...
        Train_beh = copy.deepcopy(Train[var_group])
        Val_beh = copy.deepcopy(Val[var_group])
        Test_beh = copy.deepcopy(Test[var_group])
    elif isinstance(var_group,list): # if var_group is multiple variables...
        Train_beh = copy.deepcopy(Train[var_group[0]])
        Val_beh = copy.deepcopy(Val[var_group[0]])
        Test_beh = copy.deepcopy(Test[var_group[0]])
        for i in range(1,len(var_group)):
            Train_beh = [np.vstack((tb,v)) for tb,v in zip(Train_beh,Train[var_group[i]])]
            Val_beh = [np.vstack((tb,v)) for tb,v in zip(Val_beh,Val[var_group[i]])]
            Test_beh = [np.vstack((tb,v)) for tb,v in zip(Test_beh,Test[var_group[i]])]
    else:
        raise Exception('Unexpected type for var_group.')
    
    Train_b['behavior'] = Train_beh
    Val_b['behavior'] = Val_beh
    Test_b['behavior'] = Test_beh

    return Train_b, Val_b, Test_b
    
################################################################################
    
def load_config(dataset):

    """
    Load config file for a particular dataset.

    Inputs
    ------
    dataset: string containing filename of config file (without .yml extension)

    Outputs
    ------
    config: dictionary containing settings related to the dataset for each method
   
    """

    # Load config file.
    config_path = 'config/' + dataset + '.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

################################################################################

def relevant_config(config, method, var_group):

    """
    Restrict config dictionary to the settings specific to a given method and variable group(s).

    Inputs
    ------
    config: dictionary containing settings related to the dataset for each method

    method: string specifying which method to get settings for

    var_group: string (or list of strings) containing behavioral variable group(s) to get settings for

    Outputs
    -------
    model_config: dictionary containing settings from config that are relevant
        to the specified method and variable group(s)
   
    """

    # Get the portion of the config dictionary relevant to the provided method and variable group.
    if isinstance(var_group,str):
        model_config = {key:config[method][key] for key in ['general','opt',var_group]}
    elif isinstance(var_group,list):
        vg_name = '-'.join(var_group)
        model_config = {key:config[method][key] for key in ['general','opt',vg_name]}
    else:
        raise Exception('Unexpected type for var_group.')
    
    return model_config

################################################################################

def store_results(R2, behavior, behavior_estimate, HyperParams, Results, var_group, Train):

    """
    Store coefficient of determination (R2), ground truth behavior, 
    decoed behavior, and hyperparameters in the Results dictionary
    under key(s) indicating the behavioral variable group(s) these
    results correspond to.

    Inputs
    ------
    R2: 1D numpy array of coefficients of determination

    behavior: list of M x T numpy arrays, each of which contains ground truth behavioral data for M behavioral variables over T times

    behavior_estimate: list of M x T numpy arrays, each of which contains decoded behavioral data for M behavioral variables over T times

    HyperParams: dictionary of hyperparameters

    Results: method- and dataset-specific dictionary to store results in

    var_group: string (or list of strings) containing behavioral variable group(s)
        these results are associated with

    Train: dictionary containing trialized neural and behavioral data in training set
        This only gets used to help determine which R2 values go with which behavioral variables.
   
    """

    # Store R2, behavior, decoded behavior, and HyperParams in Results with the appropriate key.
    if isinstance(var_group,str):
        Results[var_group] = dict()
        Results[var_group]['R2'] = R2
        Results[var_group]['behavior'] = behavior
        Results[var_group]['behavior_estimate'] = behavior_estimate
        Results[var_group]['HyperParams'] = HyperParams.copy()
    elif isinstance(var_group,list):
        i = 0
        for v in var_group:
            m = Train[v][0].shape[0]
            Results[v] = dict()
            Results[v]['R2'] = R2[i:i+m]
            Results[v]['behavior'] = [b[i:i+m,:] for b in behavior]
            Results[v]['behavior_estimate'] = [b[i:i+m,:] for b in behavior_estimate]
            Results[v]['HyperParams'] = HyperParams.copy()
            i += m
    else:
        raise Exception('Unexpected type for var_group.')
    
    