import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random
from sklearn.metrics import balanced_accuracy_score

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
        self.metric = 'r2'        
        self.session_active_neurons = {}    
        self.avail_mod = kwargs.get("avail_mod", None)
        self.avail_beh = kwargs.get("avail_beh", None)
        self.modal_filter = kwargs.get("modal_filter", None)
        self.mod_to_indx = {r: i for i,r in enumerate(self.avail_mod)}

        # Multi-task-Masing (MtM)
        if self.config.training.mask_type == "input":
            # self.masking_schemes = ['inter-region', 'intra-region', 'neuron', 'temporal']
            self.masking_schemes = self.config.training.mask_mode
        else:
            self.masking_mode = None

        self.mixed_training = kwargs.get("mixed_training", False)
        if self.mixed_training:
            self.training_schemes = [
                'encoding', 'decoding', 'spike-spike', 'behavior-behavior', 'token_masking'
            ]
        else:
            self.training_mode = None

    
    def _forward_model_outputs(self, batch, masking_mode, training_mode):
        
        # single_modal = True if len(self.modal_filter['output']) == 1 else False
        single_modal = True if len(self.modal_filter['output']) ==1 or len(self.modal_filter['output']) == len(self.avail_beh) else False
        batch = move_batch_to_device(batch, self.accelerator.device)
        # print(batch.keys())
        mod_dict = {}
        for mod in self.mod_to_indx.keys():
            mod_dict[mod] = {}
            mod_dict[mod]['inputs_modality'] = torch.tensor(self.mod_to_indx[mod]).to(self.accelerator.device)
            mod_dict[mod]['targets_modality'] = torch.tensor(self.mod_to_indx[mod]).to(self.accelerator.device)
            # mod_dict[mod]['inputs_attn_mask'] = batch['time_attn_mask'][:,:1] if mod in ['choice', 'block'] else batch['time_attn_mask']
            mod_dict[mod]['inputs_attn_mask'] = batch['time_attn_mask']
            # mod_dict[mod]['inputs_timestamp'] = batch['spikes_timestamps'][:,:1] if mod in ['choice', 'block'] else batch['spikes_timestamps']
            mod_dict[mod]['inputs_timestamp'] = batch['spikes_timestamps']
            # mod_dict[mod]['targets_timestamp'] = batch['spikes_timestamps'][:,:1] if mod in ['choice', 'block'] else batch['spikes_timestamps']
            mod_dict[mod]['targets_timestamp'] = batch['spikes_timestamps']
            mod_dict[mod]['eid'] = batch['eid'][0]  # each batch is from the same eid
            mod_dict[mod]['num_neuron'] = batch['spikes_data'].shape[2]
            mod_dict[mod]['masking_mode'] = masking_mode
            mod_dict[mod]['training_mode'] = training_mode
            
            if mod == 'ap':
                mod_dict[mod]['inputs'] = batch['spikes_data'].clone()
                mod_dict[mod]['targets'] = batch['spikes_data'].clone()
                mod_dict[mod]['inputs_regions'] = np.asarray(batch['neuron_regions']).T
            elif mod in self.avail_beh:
                mod_dict[mod]['inputs'] = batch[mod].clone()
                mod_dict[mod]['targets'] = batch[mod].clone()
            else:
               raise Exception(f"Modality not implemented yet.")
            
            if single_modal and mod in self.modal_filter['output']:
                mod_dict[mod]['eval_mask'] = torch.ones_like(batch['spikes_data']).to(batch['spikes_data'].device, torch.int64)
            else:
                mod_dict[mod]['eval_mask'] = torch.zeros_like(batch['spikes_data']).to(batch['spikes_data'].device, torch.int64)

        if not single_modal:
            if training_mode == 'encoding':
                for mod in self.mod_to_indx.keys():
                    if mod == 'ap':
                        mod_dict[mod]['eval_mask'] = torch.ones_like(batch['spikes_data']).to(batch['spikes_data'].device, torch.int64)
                    else:
                        mod_dict[mod]['eval_mask'] = torch.zeros_like(batch['spikes_data']).to(batch['spikes_data'].device, torch.int64)
            elif training_mode == 'decoding':
                for mod in self.mod_to_indx.keys():
                    if mod in self.avail_beh:
                        mod_dict[mod]['eval_mask'] = torch.ones_like(batch['target']).to(batch['target'].device, torch.int64)
                    else:
                        mod_dict[mod]['eval_mask'] = torch.zeros_like(batch['target']).to(batch['target'].device, torch.int64)
            elif training_mode == 'token_masking':
                for mod in self.mod_to_indx.keys():
                    mod_dict[mod]['eval_mask'] = None
            elif training_mode == 'spike-spike':
                mod_dict['ap']['eval_mask'] = None
                for mod in self.mod_to_indx.keys():
                    if mod in self.avail_beh:
                        mod_dict[mod]['eval_mask'] = torch.zeros_like(batch['target']).to(batch['target'].device, torch.int64)
            elif training_mode == 'behavior-behavior':
                for mod in self.mod_to_indx.keys():
                    if mod in self.avail_beh:
                        mod_dict[mod]['eval_mask'] = None
                mod_dict['ap']['eval_mask'] = torch.zeros_like(batch['spikes_data']).to(batch['spikes_data'].device, torch.int64)
            else:
               raise Exception(f"Training objective not implemented yet.")

        return self.model(mod_dict)

    
    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        best_eval_avg_spike_r2 = -torch.tensor(float('inf'))
        best_eval_avg_behave_r2 = -torch.tensor(float('inf'))
        #####
        best_eval_avg_static_acc = -torch.tensor(float('inf'))
        #####
        
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:

                if eval_epoch_results[f'eval_avg_spike_r2'] > best_eval_avg_spike_r2:
                    best_eval_avg_spike_r2 = eval_epoch_results['eval_avg_spike_r2']
                    print(f"epoch: {epoch} best trial avg spike r2: {best_eval_avg_spike_r2}")
                    self.save_model(name="best_spike", epoch=epoch)
                    wandb.log({"best_spike_epoch": epoch}) if self.config.wandb.use else None

                if eval_epoch_results[f'eval_avg_behave_r2'] > best_eval_avg_behave_r2:
                    best_eval_avg_behave_r2 = eval_epoch_results['eval_avg_behave_r2']
                    print(f"epoch: {epoch} best trial avg behavior r2: {best_eval_avg_behave_r2}")
                    self.save_model(name="best_behave", epoch=epoch)
                    wandb.log({"best_behave_epoch": epoch}) if self.config.wandb.use else None

                #####
                if eval_epoch_results[f'eval_avg_static_acc'] > best_eval_avg_static_acc:
                    best_eval_avg_static_acc = eval_epoch_results['eval_avg_static_acc']
                    print(f"epoch: {epoch} best trial avg static acc: {best_eval_avg_static_acc}")
                    self.save_model(name="best_static", epoch=epoch)
                    wandb.log({"best_static_epoch": epoch}) if self.config.wandb.use else None
                #####
                
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss} trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    self.save_model(name="best", epoch=epoch)
                    # https://discuss.pytorch.org/t/saving-model-and-optimiser-and-scheduler/52030/8
                    # if len(self.eid_list) > 1:
                    #     ckpt = { 
                    #         'epoch': epoch,
                    #         'model': self.model,
                    #         'optimizer': self.optimizer,
                    #         'lr_sched': self.lr_scheduler}
                    #     torch.save(ckpt, 'ckpt.pth')

                    for mod in self.modal_filter['output']:
                        if mod in ['choice', 'block']:
                            continue
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0][mod], 
                            preds=eval_epoch_results['eval_preds'][0][mod], 
                            epoch=epoch,
                            active_neurons=next(iter(self.session_active_neurons.values()))[mod][:5],
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
                if self.config.model.use_contrastive:
                    print(f"epoch: {epoch} eval s2b acc: {eval_epoch_results['eval_s2b_acc']} eval b2s acc: {eval_epoch_results['eval_b2s_acc']}")

            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                for mod in self.modal_filter['output']:
                    if mod in ['choice', 'block']:
                        continue
                    # take the first session for plotting
                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0][mod], 
                        preds=eval_epoch_results['eval_preds'][0][mod], 
                        epoch=epoch, 
                        modality=mod,
                        active_neurons=next(iter(self.session_active_neurons.values()))[mod][:5]
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
            logs_results = {**eval_epoch_results, **train_epoch_results, "epoch": epoch}
            logs_results.pop("eval_gt", None)
            logs_results.pop("eval_preds", None)
            if self.config.wandb.use:
                wandb.log(logs_results)
            else:
                print(logs_results)
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            #####
            wandb.log({"best_eval_loss": best_eval_loss,
                       f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric,
                       "best_eval_avg_spike_r2": best_eval_avg_spike_r2,
                       "best_eval_avg_behave_r2": best_eval_avg_behave_r2,
                      }
                     )
            #####

    
    def train_epoch(self, epoch):
        train_loss = 0.
        mod_loss_dict = {f"train_{mod}_loss": 0. for mod in self.modal_filter['output']}
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.config.training.mask_type == "input":
                self.masking_mode = random.sample(self.masking_schemes, 1)[0]
            if self.mixed_training:
                self.training_mode = random.sample(self.training_schemes, 1)[0]
            outputs = self._forward_model_outputs(
                batch, masking_mode=self.masking_mode, training_mode=self.training_mode
            )
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()

            for mod in self.modal_filter['output']:
                mod_loss_dict[f"train_{mod}_loss"] += outputs.mod_loss[mod]
        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.train_dataloader)
        return{
            "train_loss": train_loss/len(self.train_dataloader),
            **mod_loss_dict
        }
    
    
    def eval_epoch(self):
        
        self.model.eval()
        eval_loss = 0.
        session_results = {}
        for eid in self.eid_list:
            session_results[eid] = {}
            for mod in self.modal_filter['output']:
                session_results[eid][mod] = {"gt": [], "preds": []}
                #####
            session_results[eid]['s2b_acc'] = []
            session_results[eid]['b2s_acc'] = []
        mod_loss_dict = {f"eval_{mod}_loss": 0. for mod in self.modal_filter['output']}
        if self.eval_dataloader:

            with torch.no_grad():  

                if 'ap' in self.modal_filter['output']:
                    
                    for batch in self.eval_dataloader:
                        
                        if self.config.training.mask_type == "input":
                            self.masking_mode = random.sample(self.masking_schemes, 1)[0]
                        
                        outputs = self._forward_model_outputs(
                            batch, masking_mode=self.masking_mode, training_mode="encoding"
                        )
                        loss = outputs.loss
                        eval_loss += loss.item()
                        # eval_spike_loss += outputs.mod_loss['ap'].item()
                        mod_loss_dict[f"eval_ap_loss"] += outputs.mod_loss['ap']
                        num_neuron = batch['spikes_data'].shape[2] 
                        eid = batch['eid'][0]

                        session_results[eid]['ap']["gt"].append(
                            outputs.mod_targets['ap'].clone()[:,:,:num_neuron]
                        )
                        session_results[eid]['ap']["preds"].append(
                            outputs.mod_preds['ap'].clone()[:,:,:num_neuron]
                        )
    
                        if outputs.contrastive_dict:
                            session_results[eid]['b2s_acc'].append(outputs.contrastive_dict['b2s_acc'])
                            session_results[eid]['s2b_acc'].append(outputs.contrastive_dict['s2b_acc'])

                if 'wheel' in self.modal_filter['output']:
                    
                    for batch in self.eval_dataloader:
                        
                        if self.config.training.mask_type == "input":
                            self.masking_mode = random.sample(self.masking_schemes, 1)[0]
                        
                        outputs = self._forward_model_outputs(
                            batch, masking_mode=self.masking_mode, training_mode="decoding"
                        )
                        loss = outputs.loss
                        eval_loss += loss.item()
                        #####
                        # eval_behave_loss += outputs.mod_loss['dynamic'].item()
                        # eval_static_loss += outputs.mod_loss['static'].item()
                        #####
                        for mod in self.avail_beh:
                            mod_loss_dict[f"eval_{mod}_loss"] += outputs.mod_loss[mod]
                        num_neuron = batch['spikes_data'].shape[2] 
                        eid = batch['eid'][0]
                        for mod in self.avail_beh:
                            session_results[eid][mod]["gt"].append(
                                outputs.mod_targets[mod].clone()
                            )
                            session_results[eid][mod]["preds"].append(
                                outputs.mod_preds[mod].clone()
                            )
                        if outputs.contrastive_dict:
                            session_results[eid]['b2s_acc'].append(outputs.contrastive_dict['b2s_acc'])
                            session_results[eid]['s2b_acc'].append(outputs.contrastive_dict['s2b_acc'])

            
            gt, preds, s2b_acc_list, b2s_acc_list = {}, {}, [], []
            spike_r2_results_list, behave_r2_results_list = [], []
            #####
            gt_static, preds_static = {}, {}
            choice_acc_results_list, block_acc_results_list, acc_results_list = [], [], []
            #####
            for idx, eid in enumerate(self.eid_list):
                gt[idx], preds[idx] = {}, {}
                gt_static[idx], preds_static[idx] = {}, {}
                for mod in self.modal_filter['output']:
                    #####
                    _gt = torch.cat(session_results[eid][mod]["gt"], dim=0)
                    #####
                    _preds = torch.cat(session_results[eid][mod]["preds"], dim=0)
                    if mod == 'ap' and 'ap' in self.modal_filter['output']:
                        _preds = torch.exp(_preds)
                    gt[idx][mod] = _gt
                    preds[idx][mod] = _preds
                    
                if eid not in self.session_active_neurons:
                    self.session_active_neurons[eid] = {}
                continuous_behav_dict = {'preds': [], 'gt': []}
                for mod in self.modal_filter['output']:
                    
                    if mod == 'ap':
                        active_neurons = np.arange(gt[idx][mod].shape[-1]).tolist()
                        self.session_active_neurons[eid][mod] = active_neurons
                        
                    if mod in ['wheel', 'whisker']:
                        self.session_active_neurons[eid][mod] = [i for i in range(gt[idx][mod].size(2))]
                    
                    if mod == 'ap':
                        results = metrics_list(
                            gt = gt[idx][mod][:,:,self.session_active_neurons[eid][mod]].transpose(-1,0),
                            pred = preds[idx][mod][:,:,self.session_active_neurons[eid][mod]].transpose(-1,0), 
                            metrics=["bps"], 
                            device=self.accelerator.device
                        )
                        spike_r2_results_list.append(results["bps"])
                      
                    elif mod in ['wheel', 'whisker']:
                        continuous_behav_dict['gt'].append(gt[idx][mod])
                        continuous_behav_dict['preds'].append(preds[idx][mod])

                        #####
                    elif mod in ['choice', 'block']:
                        acc_results_list.append(balanced_accuracy_score(
                            gt[idx][mod].cpu().numpy(), preds[idx][mod].cpu().numpy()
                        ))
                        if mod == 'choice':
                            choice_acc_results_list.append(acc_results_list[-1])
                        elif mod == 'block':
                            block_acc_results_list.append(acc_results_list[-1])
                if 'wheel' in self.modal_filter['output']:
                    continuous_behav_dict['gt'] = torch.cat(continuous_behav_dict['gt'], dim=-1)
                    continuous_behav_dict['preds'] = torch.cat(continuous_behav_dict['preds'], dim=-1)
                    behave_r2_results_list.append(metrics_list(
                        gt = continuous_behav_dict['gt'],
                        pred = continuous_behav_dict['preds'],
                        metrics=["rsquared"],
                        device=self.accelerator.device
                    )["rsquared"])
                    
                if self.config.model.use_contrastive:
                    assert len(session_results[eid]['s2b_acc']) == len(session_results[eid]['b2s_acc'])
                    assert len(session_results[eid]['s2b_acc']) > 0
                    s2b_acc_list.append(np.mean(session_results[eid]['s2b_acc']))
                    b2s_acc_list.append(np.mean(session_results[eid]['b2s_acc']))
                else:
                    s2b_acc_list = [0]
                    b2s_acc_list = [0]

        spike_r2 = np.nanmean(spike_r2_results_list)
        behave_r2 = np.nanmean(behave_r2_results_list)
        choice_acc = np.nanmean(choice_acc_results_list)
        block_acc = np.nanmean(block_acc_results_list)
        acc = np.nanmean(acc_results_list)
        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.eval_dataloader)
        return {
            "eval_loss": eval_loss/len(self.eval_dataloader),
            **mod_loss_dict,
            f"eval_trial_avg_{self.metric}": np.nanmean([spike_r2, behave_r2, choice_acc, block_acc]),
             #####
            "eval_avg_spike_r2": spike_r2,
            "eval_avg_behave_r2": behave_r2,
            #####
            "eval_avg_static_acc": acc,
            "eval_avg_choice_acc": choice_acc,
            "eval_avg_block_acc": block_acc,
            #####
            "eval_gt": gt,
            "eval_preds": preds,
            "eval_s2b_acc": np.mean(s2b_acc_list),
            "eval_b2s_acc": np.mean(b2s_acc_list)
        }

    
    def plot_epoch(self, gt, preds, epoch, active_neurons, modality):
        
        if modality == 'ap':
            gt_pred_fig = plot_gt_pred(
                gt = gt.mean(0).T.cpu().numpy(),
                pred = preds.mean(0).T.detach().cpu().numpy(),
                epoch = epoch,
                modality = modality
                )
        elif modality in ['wheel', 'whisker']:
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
        