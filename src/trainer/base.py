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
from sklearn.metrics import balanced_accuracy_score, r2_score

OUTPUT_DIM = {
    "choice": 2, "block": 3, "wheel": 1, "whisker": 1, 
    "finger_x_vel": 1, "finger_y_vel": 1
}

def set_seed(epoch, base_seed=42):
    seed = base_seed + epoch
    print("Train seed set to {}.".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
        self.multi_gpu = kwargs.get("multi_gpu", None)

        self.model_class = self.config.model.model_class
        self.session_active_neurons = {}   
        if self.multi_gpu:
            self.mod_to_indx = self.model.module.mod_to_indx
        else:
            self.mod_to_indx = self.model.mod_to_indx
        self.avail_mod = kwargs.get("avail_mod", None)
        self.avail_beh = kwargs.get("avail_beh", None)
        self.modal_filter = kwargs.get("modal_filter", None)

        self.n_output_mods = len(self.modal_filter["output"])

        self.mixed_training = kwargs.get("mixed_training", False)
        self.zero_shot_transfer = kwargs.get("zero_shot_transfer", False)

        if self.mixed_training:
            self.training_mode = "mixed"
        elif self.zero_shot_transfer:
            self.training_mode = "self-spike"
            self.training_schemes = ["self-spike"]
        else:
            self.training_schemes = [
                "encoding", "decoding", 
                "self-spike", "self-behavior", 
                "random_token"
            ]

        self.enc_task_var = kwargs.get("enc_task_var", False)

        self.start_epoch = kwargs.get("start_epoch", 0)

        self.STATIC_VARS = ["choice", "block"] if "choice" in self.avail_beh else ["finger_x_vel", "finger_y_vel"]
        self.DYNAMIC_VARS = ["wheel", "whisker"] if "wheel" in self.avail_beh else []


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
    
    def _forward_model_inputs(self, batch, training_mode, enc_task_var=None):
        
        is_unimodal = True if self.n_output_mods in [1, len(self.avail_beh)] else False
        is_multimodal = not is_unimodal
        
        batch = move_batch_to_device(batch, self.accelerator.device)

        all_ones = torch.ones_like(
            batch["spikes_data"]).to(self.accelerator.device, torch.int64
        )
        all_zeros = all_ones * 0.
        
        mod_dict = {}

        avail_mod = self.mod_to_indx.keys() if not self.zero_shot_transfer else self.modal_filter["output"]
        
        for mod in avail_mod:
            
            mod_idx = self.mod_to_indx[mod]
            mod_dict[mod] = {}
            mod_dict[mod]["inputs_modality"] = torch.tensor(mod_idx).to(self.accelerator.device)
            mod_dict[mod]["targets_modality"] = torch.tensor(mod_idx).to(self.accelerator.device)
            mod_dict[mod]["inputs_attn_mask"] = batch["time_attn_mask"]
            mod_dict[mod]["inputs_timestamp"] = batch["spikes_timestamps"]
            mod_dict[mod]["targets_timestamp"] = batch["spikes_timestamps"]
            # Each batch contains samples from different sessions
            mod_dict[mod]["eid"] = batch["eid"]
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

        # Mask randomly selected modalities for encoding
        if enc_task_var is not None and enc_task_var != "all":
            for mod in self.mod_to_indx.keys():
                mod_dict[mod]["inputs_token_mask"] = all_zeros if mod == enc_task_var else all_ones

        return self.model(mod_dict)

    def _plot_log_epoch(self, epoch, eval_epoch_results, n_viz=5):
        
        for mod in self.modal_filter["output"]:
            if mod in self.STATIC_VARS:
                continue
            gt_pred_fig = self.plot_epoch(
                gt=eval_epoch_results["eval_gt"][0][mod], 
                preds=eval_epoch_results["eval_preds"][0][mod], 
                epoch=epoch,
                active_neurons=next(iter(self.session_active_neurons.values()))[mod][:n_viz],
                modality=mod
            )
            if self.config.wandb.use:
                if self.accelerator.is_main_process:
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

        if "spike" in self.modal_filter["output"] and self.enc_task_var == "random":
            best_eval_enc_metric = {
                f"eval_enc_{enc_task_var}_metric": - MAX_VAL for enc_task_var in self.STATIC_VARS + self.DYNAMIC_VARS
            }
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            
            train_epoch_results = self.train_epoch(epoch)

            eval_every = self.config.training.eval_every

            if self.accelerator.is_main_process and epoch % eval_every == 0:

                eval_epoch_results = self.eval_epoch()
                print(f"Epoch: {epoch} train loss: {train_epoch_results['train_loss']}")
                print(f"Epoch: {epoch} val loss: {eval_epoch_results['eval_loss']} val metric: {eval_epoch_results['eval_avg_metric']}")

                if eval_epoch_results:

                    for eval_name in best_eval_metric.keys():
                        if "finger_x_vel" in self.modal_filter["output"]:
                            mode = "_".join(eval_name.split("_")[1:])
                        else:
                            mode = eval_name.split("_")[1]
                        if eval_epoch_results[eval_name] > best_eval_metric[eval_name]:
                            best_eval_metric[eval_name] = eval_epoch_results[eval_name]
                            print(
                                f"Epoch: {epoch} best val {mode} metric: {best_eval_metric[eval_name]}"
                            )
                            self.save_model(name=f"best_{mode}", epoch=epoch)

                            if self.config.wandb.use:
                                wandb.log({f"best_{mode}_epoch": epoch}) if self.config.wandb.use else None
                    
                    if eval_epoch_results["eval_avg_metric"] > best_eval_metric["eval_avg_metric"]:
                        best_eval_loss = eval_epoch_results["eval_loss"]
                        best_eval_metric["eval_avg_metric"] = eval_epoch_results["eval_avg_metric"]
                        print(
                            f"Epoch: {epoch} best val loss: {best_eval_loss} val metric: {best_eval_metric['eval_avg_metric']}"
                        )
                        self.save_model(name="best", epoch=epoch)
                        self._plot_log_epoch(epoch, eval_epoch_results)
                        if self.config.wandb.use:
                            wandb.log({"best_epoch": epoch})    

                if "spike" in self.modal_filter["output"] and self.enc_task_var == "random":
                    eval_enc_results = self.eval_enc_epoch() 

                    for eval_name in best_eval_enc_metric.keys():
                        enc_task_var = eval_name.split("_")[2]
                        if eval_enc_results[eval_name] > best_eval_enc_metric[eval_name]:
                            best_eval_enc_metric[eval_name] = eval_enc_results[eval_name]
                            print(
                                f"Epoch: {epoch} best val enc {enc_task_var} metric: {best_eval_enc_metric[eval_name]}"
                            )
                            self.save_model(name=f"best_enc_{enc_task_var}", epoch=epoch)
                            if self.config.wandb.use:
                                wandb.log(eval_enc_results) 
                        
                if epoch % self.config.training.save_plot_every_n_epochs == 0:
                    self._plot_log_epoch(epoch, eval_epoch_results)

                logs_results = {"epoch": epoch, **train_epoch_results, **eval_epoch_results}
                logs_results.pop("eval_gt", None)
                logs_results.pop("eval_preds", None)

                if self.config.wandb.use:
                    wandb.log(logs_results)
                else:
                    print(logs_results)

            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)
                
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            if self.accelerator.is_main_process:
                wandb.log({"best_eval_loss": best_eval_loss, **best_eval_metric})

        return best_eval_metric

    
    def train_epoch(self, epoch):
        train_loss = 0.
        mod_loss_dict = {f"train_{mod}_loss": 0. for mod in self.modal_filter["output"]}

        set_seed(epoch)
        
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            
            if not self.mixed_training:
                self.training_mode = random.sample(self.training_schemes, 1)[0]
                
            if self.training_mode == "encoding":
                if self.enc_task_var == "random":
                    enc_task_var = random.sample(self.STATIC_VARS+self.DYNAMIC_VARS+["all"], 1)[0]
                else:
                    enc_task_var = self.enc_task_var
            else:
                enc_task_var = None

            outputs = self._forward_model_inputs(batch, self.training_mode, enc_task_var)
            loss = outputs.loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            
            train_loss += loss.item()

            for mod in self.modal_filter["output"]:
                mod_loss_dict[f"train_{mod}_loss"] += outputs.mod_loss[mod]

        print(f"Epoch {epoch} LR: {self.lr_scheduler.get_last_lr()}")
                
        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.train_dataloader)
            
        return{"train_loss": train_loss/len(self.train_dataloader), **mod_loss_dict}

    
    def _collect_eval_results(self, session_results, eval_loss, mod_loss_dict):
        self.model.eval()
        
        if self.eval_dataloader:
            with torch.no_grad(): 
                    
                if "spike" in self.modal_filter["output"]:
                    for batch in self.eval_dataloader:
                        eid = np.array(batch["eid"])
                        space_attn_mask = batch["space_attn_mask"]
                        enc_task_var = "all" if self.enc_task_var in ["all", "random"] else self.enc_task_var
                        outputs = self._forward_model_inputs(
                            batch, training_mode="encoding", enc_task_var=enc_task_var
                        )
                        eval_loss += outputs.loss.item()
                        mod_loss_dict["eval_spike_loss"] += outputs.mod_loss["spike"]
                        unique_eids = np.unique(eid)
                        for group_eid in unique_eids:
                            mask = np.argwhere(eid == group_eid).squeeze()
                            if mask.size == 0 or mask.ndim == 0:  
                                num_neuron = 0
                            else:  
                                num_neuron = torch.sum(space_attn_mask[mask][0] != 0).item()
                            if num_neuron > 0:
                                _gt = outputs.mod_targets["spike"][mask,:,:num_neuron]
                                _pred = outputs.mod_preds["spike"][mask,:,:num_neuron]
                                if len(mask) == 1:
                                    _gt = _gt.unsqueeze(0)
                                    _pred = _pred.unsqueeze(0)
                                session_results[group_eid]["spike"]["gt"].append(_gt)
                                session_results[group_eid]["spike"]["preds"].append(_pred)
    
                if ("wheel" in self.modal_filter["output"]) or ("finger_x_vel" in self.modal_filter["output"]):
                    for batch in self.eval_dataloader:
                        eid = np.array(batch["eid"])
                        outputs = self._forward_model_inputs(batch, training_mode="decoding")
                        eval_loss += outputs.loss.item()
                        for mod in self.avail_beh:
                            mod_loss_dict[f"eval_{mod}_loss"] += outputs.mod_loss[mod]     
                            unique_eids = np.unique(eid)
                            for group_eid in unique_eids:
                                mask = np.argwhere(eid == group_eid).squeeze()
                                if mask.size == 0 or mask.ndim == 0:  
                                    continue
                                else: 
                                    _gt = outputs.mod_targets[mod][mask]
                                    _pred = outputs.mod_preds[mod][mask]
                                    if len(mask) == 1:
                                        _gt = _gt.unsqueeze(0)
                                        _pred = _pred.unsqueeze(0)
                                    session_results[group_eid][mod]["gt"].append(_gt)
                                    session_results[group_eid][mod]["preds"].append(_pred)

        return session_results, eval_loss, mod_loss_dict
    

    def _collect_enc_results(self, session_enc_results):
        self.model.eval()
        
        if self.eval_dataloader:
            with torch.no_grad(): 
                for enc_task_var in self.STATIC_VARS + self.DYNAMIC_VARS:
                    for batch in self.eval_dataloader:
                        eid = np.array(batch["eid"])
                        space_attn_mask = batch["space_attn_mask"]
                        outputs = self._forward_model_inputs(
                            batch, training_mode="encoding", enc_task_var=enc_task_var
                        )
                        unique_eids = np.unique(eid)
                        for group_eid in unique_eids:
                            mask = np.argwhere(eid == group_eid).squeeze()
                            if mask.size == 0 or mask.ndim == 0:  
                                num_neuron = 0
                            else:  
                                num_neuron = torch.sum(space_attn_mask[mask][0] != 0).item()
                            if num_neuron > 0:
                                _gt = outputs.mod_targets["spike"][mask,:,:num_neuron]
                                _pred = outputs.mod_preds["spike"][mask,:,:num_neuron]
                                if len(mask) == 1:
                                    _gt = _gt.unsqueeze(0)
                                    _pred = _pred.unsqueeze(0)
                                session_enc_results[group_eid][enc_task_var]["gt"].append(_gt)
                                session_enc_results[group_eid][enc_task_var]["preds"].append(_pred)

        return session_enc_results
    

    def eval_enc_epoch(self):

        session_enc_results = {}
        for eid in self.eid_list:
            session_enc_results[eid] = {}
            for enc_task_var in self.STATIC_VARS + self.DYNAMIC_VARS:
                session_enc_results[eid][enc_task_var] = {"gt": [], "preds": []}

        session_enc_results = self._collect_enc_results(session_enc_results)

        gt, preds, eval_metrics = {}, {}, {enc_task_var: [] for enc_task_var in self.STATIC_VARS + self.DYNAMIC_VARS}
        for idx, eid in enumerate(self.eid_list):
            gt[idx], preds[idx] = {}, {}
            for enc_task_var in self.STATIC_VARS + self.DYNAMIC_VARS:
                _gt = torch.cat(session_enc_results[eid][enc_task_var]["gt"], dim=0)
                _preds = torch.cat(session_enc_results[eid][enc_task_var]["preds"], dim=0)
                _preds = torch.exp(_preds)
                if _gt.ndim == 2 and _preds.ndim == 2:
                    _gt, _preds = _gt.unsqueeze(0), _preds.unsqueeze(0)
                gt[idx][enc_task_var], preds[idx][enc_task_var] = _gt, _preds

                results = metrics_list(
                    gt = gt[idx][enc_task_var].transpose(-1,0), 
                    pred = preds[idx][enc_task_var].transpose(-1,0), 
                    metrics=["bps"], 
                    device=self.accelerator.device
                )
                eval_metrics[enc_task_var].append(results["bps"])

        enc_task_var_metric_dict = {}
        for enc_task_var in eval_metrics.keys():
            enc_task_var_metric_dict[f"eval_enc_{enc_task_var}_metric"] = np.nanmean(eval_metrics[enc_task_var])
            
        return enc_task_var_metric_dict
    
    
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
                try:
                    _gt = torch.cat(session_results[eid][mod]["gt"], dim=0)
                    _preds = torch.cat(session_results[eid][mod]["preds"], dim=0)
                except:
                    print(f"Missing EID {idx}: {eid} Modality: {mod}")
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
                
                elif mod in self.DYNAMIC_VARS:
                    self.session_active_neurons[eid][mod] = [i for i in range(gt[idx][mod].size(-1))]
                    results = metrics_list(
                        gt = gt[idx][mod].unsqueeze(-1) if gt[idx][mod].ndim == 2 else gt[idx][mod], 
                        pred = preds[idx][mod].unsqueeze(-1) if preds[idx][mod].ndim == 2 else preds[idx][mod],
                        metrics=["behave_r2"], device=self.accelerator.device
                    )
                    results["behave_r2"] = np.nan if results["behave_r2"] == -float("inf") else results["behave_r2"]
                    eval_metrics[mod].append(results["behave_r2"])
                
                elif mod in self.STATIC_VARS:
                    try:
                        if mod in ["choice", "block"]:
                            metric = balanced_accuracy_score(
                                gt[idx][mod].cpu().numpy(), preds[idx][mod].cpu().numpy()
                            )
                        else:
                            metric = r2_score(
                                gt[idx][mod].cpu().numpy(), preds[idx][mod].cpu().numpy()
                            )
                    except ValueError:
                        metric = np.nan
                    eval_metrics[mod].append(metric)

        for key in mod_loss_dict.keys():
            mod_loss_dict[key] /= len(self.eval_dataloader)

        mod_metric_dict = {}
        for mod in eval_metrics.keys():
            mod_metric_dict[f"eval_{mod}_metric"] = np.nanmean(eval_metrics[mod])

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
        elif modality in self.DYNAMIC_VARS:
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
        if self.accelerator.is_main_process:
            print(f"Saving model: {name} to {self.log_dir}")
            if self.multi_gpu:
                dict_config = {
                    "epoch": epoch,
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_sched": self.lr_scheduler.state_dict(),
                }
            else:
                dict_config = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_sched": self.lr_scheduler.state_dict(),
                }
            torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
