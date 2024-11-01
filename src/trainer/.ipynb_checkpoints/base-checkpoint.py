import os
import random
import numpy as np
from tqdm import tqdm
import wandb
import torch
from utils.utils import (
    move_batch_to_device, 
    metrics_list, 
    plot_gt_pred, 
    plot_neurons_r2
)
from sklearn.metrics import balanced_accuracy_score

STATIC_VARS = ["choice", "block"]
DYNAMIC_VARS = ["wheel", "whisker"]
OUTPUT_DIM = {"choice": 2, "block": 3, "wheel": 1, "whisker": 1}


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
        self.eid_list = kwargs.get("eid_list", None)

        self.model_class = self.config.model.model_class
        self.session_active_neurons = {}   
        self.mod_to_indx = self.model.mod_to_indx
        self.avail_mod = kwargs.get("avail_mod", None)
        self.avail_beh = kwargs.get("avail_beh", None)
        self.modal_filter = kwargs.get("modal_filter", None)

        self.n_output_mods = len(self.modal_filter["output"])

        self.mixed_training = kwargs.get("mixed_training", False)
        if self.mixed_training:
            self.training_mode = "mixed"
        else:
            self.training_schemes = [
                "encoding", "decoding", 
                "self-spike", "self-behavior", 
                "random_token"
            ]

    def _prepare_multimodal_mask(self, mod_dict, training_mode, all_ones, all_zeros):
        
        if training_mode == "encoding":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = all_ones if mod == "spike" else all_zeros
                
        elif training_mode == "decoding":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = all_ones if mod in self.avail_beh else all_zeros
                    
        elif training_mode == "random_token":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = None
                
        elif training_mode == "self-spike":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = None if mod == "spike" else all_zeros
                    
        elif training_mode == "self-behavior":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = None if mod in self.avail_beh else all_zeros
        elif training_mode == "mixed":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["eval_mask"] = None
        else:
           raise Exception(f"masking mode {training_mode} not implemented.")
                
        return mod_dict
    
    def _forward_model_inputs(self, batch, training_mode):
        
        is_unimodal = True if self.n_output_mods in [1, len(self.avail_beh)] else False
        is_multimodal = not is_unimodal
        
        batch = move_batch_to_device(batch, self.accelerator.device)

        all_ones = torch.ones_like(
            batch["spikes_data"]).to(self.accelerator.device, torch.int64
        )
        all_zeros = all_ones * 0.
        
        mod_dict = {}
        for mod in self.mod_to_indx.keys():
            
            mod_idx = self.mod_to_indx[mod]
            mod_dict[mod] = {}
            mod_dict[mod]["inputs_modality"] = torch.tensor(mod_idx).to(self.accelerator.device)
            mod_dict[mod]["targets_modality"] = torch.tensor(mod_idx).to(self.accelerator.device)
            mod_dict[mod]["inputs_attn_mask"] = batch["time_attn_mask"]
            mod_dict[mod]["inputs_timestamp"] = batch["spikes_timestamps"]
            mod_dict[mod]["targets_timestamp"] = batch["spikes_timestamps"]
            mod_dict[mod]["eid"] = batch["eid"][0]  # Each batch is from the same EID now
            mod_dict[mod]["num_neuron"] = batch["spikes_data"].shape[-1]
            mod_dict[mod]["training_mode"] = training_mode
            
            if mod == "spike":
                mod_dict[mod]["inputs"] = batch["spikes_data"].clone()
                mod_dict[mod]["targets"] = batch["spikes_data"].clone()
            elif mod in self.avail_beh:
                mod_dict[mod]["inputs"] = batch[mod].clone()
                mod_dict[mod]["targets"] = batch[mod].clone()
            else:
               raise Exception(f"modality {mod} not implemented.")
            
            mod_dict[mod]["eval_mask"] = all_ones \
            if is_unimodal and mod in self.modal_filter["output"] else all_zeros

        if is_multimodal:
            self._prepare_multimodal_mask(mod_dict, training_mode, all_ones, all_zeros)

        return self.model(mod_dict)

    def _plot_log_epoch(self, epoch, eval_epoch_results, n_viz=5):
        
        for mod in self.modal_filter["output"]:
            if mod in STATIC_VARS:
                continue
            gt_pred_fig = self.plot_epoch(
                gt=eval_epoch_results["eval_gt"][0][mod], 
                preds=eval_epoch_results["eval_preds"][0][mod], 
                epoch=epoch,
                active_neurons=next(iter(self.session_active_neurons.values()))[mod][:n_viz],
                modality=mod
            )
            if self.config.wandb.use:
                wandb.log({
                    f"best_gt_pred_fig_{mod}": wandb.Image(gt_pred_fig["plot_gt_pred"]),
                    f"best_r2_fig_{mod}": wandb.Image(gt_pred_fig["plot_r2"])
                })        

    def train(self):

        MAX_VAL = torch.tensor(float("inf"))
        best_eval_loss = MAX_VAL
        best_eval_metric = {
            f"eval_{mode}_metric": - MAX_VAL for mode in self.modal_filter["output"] + ["avg"]
        }
        
        for epoch in range(self.config.training.num_epochs):
            
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"Epoch: {epoch} train loss: {train_epoch_results['train_loss']}")
            print(f"Epoch: {epoch} val loss: {eval_epoch_results['eval_loss']} val metric: {eval_epoch_results['eval_avg_metric']}")

            if eval_epoch_results:

                for eval_name in best_eval_metric.keys():
                    mode = eval_name.split("_")[1]
                    if eval_epoch_results[eval_name] > best_eval_metric[eval_name]:
                        best_eval_metric[eval_name] = eval_epoch_results[eval_name]
                        print(
                            f"Epoch: {epoch} best val {mode} metric: {best_eval_metric[eval_name]}"
                        )
                        self.save_model(name=f"best_{mode}", epoch=epoch)
                        wandb.log({f"best_{mode}_epoch": epoch}) if self.config.wandb.use else None
                
                if eval_epoch_results["eval_avg_metric"] > best_eval_metric["eval_avg_metric"]:
                    best_eval_loss = eval_epoch_results["eval_loss"]
                    best_eval_metric["eval_avg_metric"] = eval_epoch_results["eval_avg_metric"]
                    print(f"Epoch: {epoch} best val loss: {best_eval_loss} val metric: {best_eval_metric['eval_avg_metric']}"
                    )
                    self.save_model(name="best", epoch=epoch)
                    self._plot_log_epoch(epoch, eval_epoch_results)
                    if self.config.wandb.use:
                        wandb.log({"best_epoch": epoch})        
                    
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                self._plot_log_epoch(epoch, eval_epoch_results)
                        
            logs_results = {"epoch": epoch, **train_epoch_results, **eval_epoch_results}
            logs_results.pop("eval_gt", None)
            logs_results.pop("eval_preds", None)
            if self.config.wandb.use:
                wandb.log(logs_results)
            else:
                print(logs_results)
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_eval_loss": best_eval_loss, **best_eval_metric})

    
    def train_epoch(self, epoch):
        train_loss = 0.
        mod_loss_dict = {f"train_{mod}_loss": 0. for mod in self.modal_filter["output"]}
        
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if not self.mixed_training:
                self.training_mode = random.sample(self.training_schemes, 1)[0]
            print("training_mode: ", self.training_mode)
            outputs = self._forward_model_inputs(batch, self.training_mode)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()

            for mod in self.modal_filter["output"]:
                mod_loss_dict[f"train_{mod}_loss"] += outputs.mod_loss[mod]
                
        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.train_dataloader)
            
        return{"train_loss": train_loss/len(self.train_dataloader), **mod_loss_dict}

    
    def _collect_eval_results(self, session_results, eval_loss, mod_loss_dict):
        self.model.eval()
        
        if self.eval_dataloader:
            with torch.no_grad(): 
                for batch in self.eval_dataloader:
                    num_neuron, eid = batch["spikes_data"].shape[-1], batch["eid"][0]
                    
                if "spike" in self.modal_filter["output"]:
                    for batch in self.eval_dataloader:
                        outputs = self._forward_model_inputs(batch, training_mode="encoding")
                        eval_loss += outputs.loss.item()
                        mod_loss_dict["eval_spike_loss"] += outputs.mod_loss["spike"]
                        session_results[eid]["spike"]["gt"].append(
                            outputs.mod_targets["spike"][...,:num_neuron]
                        )
                        session_results[eid]["spike"]["preds"].append(
                            outputs.mod_preds["spike"][...,:num_neuron]
                        )
    
                if "wheel" in self.modal_filter["output"]:
                    for batch in self.eval_dataloader:
                        outputs = self._forward_model_inputs(batch, training_mode="decoding")
                        eval_loss += outputs.loss.item()
                        for mod in self.avail_beh:
                            mod_loss_dict[f"eval_{mod}_loss"] += outputs.mod_loss[mod]                        
                            session_results[eid][mod]["gt"].append(outputs.mod_targets[mod].clone())
                            session_results[eid][mod]["preds"].append(outputs.mod_preds[mod].clone())
        return session_results, eval_loss, mod_loss_dict
    
    
    def eval_epoch(self):
        eval_loss = 0.
        mod_loss_dict = {f"eval_{mod}_loss": 0. for mod in self.modal_filter["output"]}
        
        session_results = {}
        for eid in self.eid_list:
            session_results[eid] = {}
            for mod in self.modal_filter["output"]:
                session_results[eid][mod] = {"gt": [], "preds": []}

        session_results, eval_loss, mod_loss_dict = self._collect_eval_results(
            session_results, eval_loss, mod_loss_dict
        )
            
        gt, preds, eval_metrics = {}, {}, {mod: [] for mod in self.modal_filter["output"]}
        for idx, eid in enumerate(self.eid_list):
            gt[idx], preds[idx] = {}, {}
            for mod in self.modal_filter["output"]:
                _gt = torch.cat(session_results[eid][mod]["gt"], dim=0)
                _preds = torch.cat(session_results[eid][mod]["preds"], dim=0)
                if mod == "spike" and "spike" in self.modal_filter["output"]:
                    _preds = torch.exp(_preds)
                gt[idx][mod], preds[idx][mod] = _gt, _preds
                
            if eid not in self.session_active_neurons:
                self.session_active_neurons[eid] = {}
                
            for mod in self.modal_filter["output"]:
                if mod == "spike":
                    self.session_active_neurons[eid][mod] = np.arange(gt[idx][mod].shape[-1]).tolist()
                    results = metrics_list(
                        gt = gt[idx][mod].transpose(-1,0), pred = preds[idx][mod].transpose(-1,0), 
                        metrics=["bps"], device=self.accelerator.device
                    )
                    eval_metrics[mod].append(results["bps"])
                
                elif mod in DYNAMIC_VARS:
                    self.session_active_neurons[eid][mod] = [i for i in range(gt[idx][mod].size(-1))]
                    results = metrics_list(
                        gt = gt[idx][mod].unsqueeze(-1), pred = preds[idx][mod].unsqueeze(-1),
                        metrics=["rsquared"], device=self.accelerator.device
                    )
                    eval_metrics[mod].append(results["rsquared"])
                
                elif mod in STATIC_VARS:
                    acc = balanced_accuracy_score(
                        gt[idx][mod].cpu().numpy(), preds[idx][mod].cpu().numpy()
                    )
                    for mod in STATIC_VARS:
                        eval_metrics[mod].append(acc)

        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.eval_dataloader)

        mod_metric_dict = {}
        for mod in eval_metrics.keys():
            mod_metric_dict[f"eval_{mod}_metric"] = np.nanmean(eval_metrics[mod])

        print(mod_metric_dict)
        mod_metric_dict["eval_avg_metric"] = np.nanmean(list(mod_metric_dict.values()))
            
        return {
            "eval_loss": eval_loss/len(self.eval_dataloader),
            **mod_loss_dict, 
            **mod_metric_dict,
            "eval_gt": gt,
            "eval_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        
        if modality == "spike":
            gt_pred_fig = plot_gt_pred(
                gt = gt.mean(0).T.cpu().numpy(),
                pred = preds.mean(0).T.detach().cpu().numpy(),
                epoch = epoch,
                modality = modality
            )
        elif modality in DYNAMIC_VARS:
            gt_pred_fig = plot_gt_pred(
                gt = gt.mean(0).T.cpu().numpy(),
                pred = preds.mean(0).T.detach().cpu().numpy(),
                epoch = epoch,
                modality=modality
            )
            active_neurons = range(gt.size()[-1])
            
        r2_fig = plot_neurons_r2(
            gt = gt.mean(0),
            pred = preds.mean(0),
            neuron_idx=active_neurons,
            epoch = epoch
        )
        return {"plot_gt_pred": gt_pred_fig, "plot_r2": r2_fig}

    def save_model(self, name="last", epoch=0):
        print(f"Saving model: {name} to {self.log_dir}")
        dict_config = {
            "epoch": epoch,
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_sched": self.lr_scheduler
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))




class BaselineTrainer():
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
  
        self.session_active_neurons = []      
        self.avail_mod = kwargs.get("avail_mod", None)
        self.modal_filter = kwargs.get("modal_filter", None)

        #####
        self.target_to_decode = kwargs["target_to_decode"]
        if ('choice' in self.target_to_decode) or ('block' in self.target_to_decode):
            self.metric = "acc"      
        else:
            self.metric = "r2"    
        #####

    def _forward_model_outputs(self, batch):
        batch = move_batch_to_device(batch, self.accelerator.device)
        data_dict = {}
        if 'ap' in self.modal_filter['output']:
            data_dict['inputs'] = batch['target']
            data_dict['targets'] = batch['spikes_data']
        else:
            data_dict['inputs'] = batch['spikes_data']
            #####
            T = len(['wheel-speed', 'whisker-motion-energy'])
            if self.target_to_decode == ['wheel-speed', 'whisker-motion-energy']:
                data_dict['targets'] = batch['target'][:,:,:T]
            elif self.target_to_decode[0] == 'choice':
                data_dict['targets'] = batch['target'][:,0,2]
            elif self.target_to_decode[0] == 'block':
                data_dict['targets'] = batch['target'][:,0,3]
            #####
        data_dict['eid'] = batch['eid'][0]  # each batch is from the same eid
        data_dict['num_neuron'] = batch['spikes_data'].shape[2]
        return self.model(data_dict)

    
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

                    for mod in self.modal_filter['output']:
                        #####
                        if ('choice' not in self.target_to_decode) and ('block' not in self.target_to_decode):
                        #####
                            gt_pred_fig = self.plot_epoch(
                                gt=eval_epoch_results['eval_gt'][0][mod], 
                                preds=eval_epoch_results['eval_preds'][0][mod], 
                                epoch=epoch,
                                active_neurons=self.session_active_neurons[0][:5], 
                                modality=mod
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
                for mod in self.modal_filter['output']:
                    #####
                    if ('choice' not in self.target_to_decode) and ('block' not in self.target_to_decode):
                    #####
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0][mod], 
                            preds=eval_epoch_results['eval_preds'][0][mod], 
                            epoch=epoch, 
                            modality=mod,
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
            outputs = self._forward_model_outputs(batch)
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
            for mod in self.modal_filter['output']:
                session_results[num_neuron][mod] = {"gt": [], "preds": []}
                
        if self.eval_dataloader:
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    outputs = self._forward_model_outputs(batch)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    num_neuron = batch['spikes_data'].shape[2] 

                    for mod in self.modal_filter['output']:
                        session_results[num_neuron][mod]["gt"].append(outputs.targets.clone())
                        session_results[num_neuron][mod]["preds"].append(outputs.preds.clone())

            gt, preds, results_list = {}, {}, []
            for idx, num_neuron in enumerate(self.num_neurons):
                gt[idx], preds[idx] = {}, {}
                for mod in self.modal_filter['output']:
                    _gt = torch.cat(session_results[num_neuron][mod]["gt"], dim=0)
                    _preds = torch.cat(session_results[num_neuron][mod]["preds"], dim=0)
                    if mod == 'ap':
                        _preds = torch.exp(_preds)
                    gt[idx][mod] = _gt
                    preds[idx][mod] = _preds
                    if mod == 'ap':
                        active_neurons = np.argsort(gt[idx][mod].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    else:
                        active_neurons = np.arange(gt[idx][mod].size(-1)).tolist()
                    self.session_active_neurons.append(active_neurons)

                for mod in self.modal_filter['output']:
                    #####
                    if ('choice' not in self.target_to_decode) and ('block' not in self.target_to_decode):
                        results = metrics_list(gt = gt[idx][mod][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                            pred = preds[idx][mod][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                            metrics=["r2"], 
                                            device=self.accelerator.device)
                        results_list.append(results["r2"])
                    else:
                        from sklearn.metrics import balanced_accuracy_score
                        results_list.append(balanced_accuracy_score(
                            gt[idx][mod].cpu().numpy(), preds[idx][mod].cpu().numpy().argmax(-1)
                        ))
                    #####

        return {
            "eval_loss": eval_loss,
            f"eval_trial_avg_{self.metric}": np.nanmean(results_list),
            "eval_gt": gt,
            "eval_preds": preds,
        }

    
    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        
        if modality == 'ap':
            gt_pred_fig = plot_gt_pred(
                gt = gt.mean(0).T.cpu().numpy(),
                pred = preds.mean(0).T.detach().cpu().numpy(),
                epoch = epoch,
                modality = modality
                )
        elif modality == 'behavior':
            gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                        pred = preds.mean(0).T.detach().cpu().numpy(),
                        epoch = epoch,
                        modality=modality)
            active_neurons = range(gt.size()[-1])
            
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
        
