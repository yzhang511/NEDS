import numpy as np
import torch
import torch.nn as nn
from torch import optim


def np2tensor(v):
    return torch.from_numpy(v)

def np2param(v, grad=True):
    return nn.Parameter(np2tensor(v), requires_grad=grad)

def tensor2np(v):
    return v.numpy()

def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device


class RRRGD():
    def __init__(self, train_data, ncomp, l2=0.):
        self.l2=l2
        self.eids = list(train_data.keys())

        np.random.seed(0)
        self.N = 0; self.model = {}
        for eid in train_data:
            _X = train_data[eid]['X'][0] # (K,T,ncoef), the last coef is the bias term
            _y = train_data[eid]['y'][0] # (K,T,N)
            K, T, ncoef = _X.shape
            K, T, N = _y.shape
            U = np.random.normal(size=(N, ncoef-1, ncomp))/np.sqrt(T*ncomp)
            V = np.random.normal(size=(ncomp, T))/np.sqrt(T*ncomp)
            b = np.expand_dims(_y.mean(0).T, 1)
            b = np.ascontiguousarray(b)
            self.model[eid+"_U"]=np2param(U)
            self.model[eid+"_b"]=np2param(b)
            self.N += N
        self.model['V'] = np2param(V) # V shared across sessions
        self.n_comp, self.T = self.model['V'].shape
        # U: model[eid+"_U"], (N, ncoef, ncomp)
        # V: model['V'], (ncomp, T)
        # b: model[eid+"_b"], (N, 1, T)

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)

    def state_dict(self):
        checkpoint = {"model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                      "l2": self.l2,
                      "eids": self.eids,
                      "N": self.N,
                      "T": self.T,
                      "n_comp": self.n_comp,}
        return checkpoint
    
    def load_state_dict(self, f):
        self.model.load_state_dict(f)

    def compute_beta(self, eid, withbias=True):
        U = self.model[eid + "_U"]
        V = self.model['V']
        b = self.model[eid + "_b"]

        beta = U @ V
        if withbias:
            beta = torch.cat((beta, b), 1) # (N, ncoef, T)
        
        return beta

    """
    - input nparray 
    - output tensor
    """
    def predict_y(self, X, y, eid):
        # X: (K, T, ncoef+1)
        # y: (K, T, N)
        beta = self.compute_beta(eid)
        X = np2tensor(X).to(beta.device)
        y = np2tensor(y).to(beta.device)
        y_pred = torch.einsum("ktc,nct->ktn", X, beta)
        return X, y, y_pred
    
    """
    - input and output nparray by default
    """
    def compute_MSE_RRRGD(self, data, k):
        mses_all = {}
        for eid in data:
            _, y, ypred = self.predict_y(data[eid]['X'][k], data[eid]['y'][k], eid)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0, 1))
        return mses_all

    def regression_loss(self):
        return {eid: self.l2*torch.sum(self.compute_beta(eid, withbias=False)**2) for eid in self.eids}

"""
train the 
    model: RRRGD_model
given the
    train_data: {eid: Xy_regression[area][eid]}
- model saved in model_fname
"""
def train_model(model, train_data, optimizer, model_fname, save=True):
    def closure():
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0;
        train_mses_all = model.compute_MSE_RRRGD(train_data, 0)
        reg_losses_all = model.regression_loss()
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
            total_loss += reg_losses_all[eid]
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    model.eval()
    mses_val = model.compute_MSE_RRRGD(train_data, 1)
    best_loss = torch.sum(torch.cat([mses_val[k] for k in mses_val]))

    if save:
        # Save the best model parameters
        checkpoint = {"RRRGD_model": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, model_fname)
        
    return model, {"mses_val":mses_val, "mse_val_mean":best_loss}

def train_model_main(train_data, l2, n_comp, model_fname,):
    area_model = RRRGD(train_data, n_comp, l2=l2)
    
    device = get_device()
    area_model.to(device)
    print(f"training on device: {device}")

    optimizer = optim.LBFGS(area_model.model.parameters(),)
    _, mse_val = train_model(area_model, train_data, optimizer,
                             model_fname=model_fname, save=True)
    return area_model, mse_val
