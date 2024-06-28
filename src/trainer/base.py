import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random


class MultiModalTrainer():
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        **kwargs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.num_neurons = kwargs.get("num_neurons", None)

        self.model_class = self.config.model.model_class
        self.metric = 'r2'        
        self.session_active_neurons = []      
        self.avail_mod = kwargs.get("avail_mod", None)
        self.mod_to_indx = {r: i for i,r in enumerate(self.avail_mod)}

        # Multi-task-Masing (MtM)
        if self.config.training.use_mtm:
            # self.masking_schemes = ['inter-region', 'intra-region', 'neuron', 'temporal']
            self.masking_schemes = ['temporal']
        else:
            self.masking_mode = None

    def _forward_model_outputs(self, batch, masking_mode):
        batch = move_batch_to_device(batch, self.accelerator.device)
        mod_dict = {}
        for mod in self.mod_to_indx.keys():
            mod_dict[mod] = {}
            mod_dict[mod]['inputs_modality'] = torch.tensor(self.mod_to_indx[mod]).to(self.accelerator.device)
            mod_dict[mod]['targets_modality'] = torch.tensor(self.mod_to_indx[mod]).to(self.accelerator.device)
            mod_dict[mod]['inputs_attn_mask'] = batch['time_attn_mask']
            mod_dict[mod]['inputs_timestamp'] = batch['spikes_timestamps']
            mod_dict[mod]['targets_timestamp'] = batch['spikes_timestamps']
            mod_dict[mod]['eid'] = batch['eid'][0]  # each batch is from the same eid
            mod_dict[mod]['num_neuron'] = batch['spikes_data'].shape[2]
            mod_dict[mod]['masking_mode'] = masking_mode
            mod_dict[mod]['eval_mask'] = None
            if mod == 'ap':
                mod_dict[mod]['inputs'] = batch['spikes_data'].clone()
                mod_dict[mod]['targets'] = batch['spikes_data'].clone()
                mod_dict[mod]['inputs_regions'] = np.asarray(batch['neuron_regions']).T
            elif mod == 'behavior':
                mod_dict[mod]['inputs'] = batch['target'].clone()
                mod_dict[mod]['targets'] = batch['target'].clone()
            else:
               raise Exception(f"Modality not implemented yet.")
        return self.model(mod_dict)

    
    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss} trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    self.save_model(name="best", epoch=epoch)

                    for mod in self.mod_to_indx.keys():
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0][mod], 
                            preds=eval_epoch_results['eval_preds'][0][mod], epoch=epoch,
                            active_neurons=self.session_active_neurons[0][:5], modality=mod
                        )
                        if self.config.wandb.use:
                            wandb.log(
                                {"best_epoch": epoch,
                                 f"best_gt_pred_fig_{mod}": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                 f"best_r2_fig_{mod}": wandb.Image(gt_pred_fig['plot_r2'])}
                            )
                        else:
                            gt_pred_fig['plot_gt_pred'].savefig(
                                os.path.join(self.log_dir, f"best_gt_pred_fig_{mod}_{epoch}.png")
                            )
                            gt_pred_fig['plot_r2'].savefig(
                                os.path.join(self.log_dir, f"best_r2_fig_{mod}_{epoch}.png")
                            )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} trial avg {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                for mod in self.mod_to_indx.keys():
                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0][mod], 
                        preds=eval_epoch_results['eval_preds'][0][mod], 
                        epoch=epoch, modality=mod,
                        active_neurons=self.session_active_neurons[0][:5]
                    )
                    if self.config.wandb.use:
                        wandb.log({
                            f"gt_pred_fig_{mod}": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            f"r2_fig_{mod}": wandb.Image(gt_pred_fig['plot_r2'])
                        })
                    else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"gt_pred_fig_{mod}_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"r2_fig_{mod}_{epoch}.png")
                        )

            if self.config.wandb.use:
                wandb.log({
                    "train_loss": train_epoch_results['train_loss'],
                    "eval_loss": eval_epoch_results['eval_loss'],
                    f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                })
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_eval_loss": best_eval_loss,
                       f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})

    
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.config.training.use_mtm:
                self.masking_mode = random.sample(self.masking_schemes, 1)[0]
            outputs = self._forward_model_outputs(batch, masking_mode=self.masking_mode)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
        return{
            "train_loss": train_loss
        }
    
    
    def eval_epoch(self):
        
        self.model.eval()
        eval_loss = 0.
        session_results = {}
        for num_neuron in self.num_neurons:
            session_results[num_neuron] = {}
            for mod in self.mod_to_indx.keys():
                session_results[num_neuron][mod] = {"gt": [], "preds": []}
                
        if self.eval_dataloader:
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.config.training.use_mtm:
                        self.masking_mode = random.sample(self.masking_schemes, 1)[0]
                    outputs = self._forward_model_outputs(batch, masking_mode=self.masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    num_neuron = batch['spikes_data'].shape[2] 

                    for mod in self.mod_to_indx.keys():
                        if mod == 'ap':
                            session_results[num_neuron][mod]["gt"].append(outputs.mod_targets[mod].clone()[:,:,:num_neuron])
                            session_results[num_neuron][mod]["preds"].append(outputs.mod_preds[mod].clone()[:,:,:num_neuron])
                        elif mod == 'behavior':
                            session_results[num_neuron][mod]["gt"].append(outputs.mod_targets[mod].clone())
                            session_results[num_neuron][mod]["preds"].append(outputs.mod_preds[mod].clone())

            gt, preds, results_list = {}, {}, []
            for idx, num_neuron in enumerate(self.num_neurons):
                gt[idx], preds[idx] = {}, {}
                for mod in self.mod_to_indx.keys():
                    _gt = torch.cat(session_results[num_neuron][mod]["gt"], dim=0)
                    _preds = torch.cat(session_results[num_neuron][mod]["preds"], dim=0)
                    if mod == 'ap':
                        _preds = torch.exp(_preds)
                    gt[idx][mod] = _gt
                    preds[idx][mod] = _preds

                if len(self.session_active_neurons) < len(self.num_neurons):
                    active_neurons = np.argsort(gt[idx]['ap'].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)

                for mod in self.mod_to_indx.keys():
                    if mod == 'ap':
                        results = metrics_list(gt = gt[idx][mod][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                            pred = preds[idx][mod][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                            metrics=["r2"], 
                                            device=self.accelerator.device)
                    elif mod == 'behavior':
                        results = metrics_list(gt = gt[idx][mod],
                                            pred = preds[idx][mod],
                                            metrics=[self.metric],
                                            device=self.accelerator.device)
                    results_list.append(results[self.metric])

        return {
            "eval_loss": eval_loss,
            f"eval_trial_avg_{self.metric}": np.nanmean(results_list),
            "eval_gt": gt,
            "eval_preds": preds,
        }

    
    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        
        if modality == 'ap':
            gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                        pred = preds.mean(0).T.detach().cpu().numpy(),
                        epoch = epoch)
        elif modality == 'behavior':
            # Hack: Enable drawing multiple behaviors later
            gt_pred_fig = plot_gt_pred(gt = gt.squeeze().cpu().numpy(),
                        pred = preds.squeeze().detach().cpu().numpy(),
                        epoch = epoch)
            active_neurons = [0]
            
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=active_neurons,
                epoch = epoch)
        
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }

    
    def save_model(self, name="last", epoch=0):
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))

