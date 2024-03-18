import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_base import AUXBase
from common import utils
from module.rl_module import RRAPredictor
from info_nce import InfoNCE

class RRA(AUXBase):

    def __init__(self, action_shape, negative_sample_num, extr_latent_dim, hidden_dim,
                extr_lr, extr_beta, device='cpu', **kwargs):
        super().__init__()
        action_dim = action_shape[0]
        # Initialize hyperparameters
        self.device = device
        self.negative_sample_num = negative_sample_num

        # Initialize modules
        self.network = RRAPredictor(extr_latent_dim,
                                   hidden_dim,
                                   action_dim).to(device)

        # Initialize optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=extr_lr, betas=(extr_beta, 0.999))
    

    def update_extr(self, data, s, s2, num_aug):
        
        # criterion = torch.nn.BCEWithLogitsLoss()
        # criterion_none_reduction = torch.nn.BCEWithLogitsLoss(reduction = 'none')
        
        with torch.no_grad():
            a = data['act']
            negative_samples = data['negative_samples'] # (batch_size, negative_sample_num, act_dim)
            r = data['rew'].squeeze() 
        if num_aug > 1:
            a = a.repeat(num_aug, 1)
            r = r.repeat(num_aug)
            negative_samples = negative_samples.repeat(num_aug, 1, 1)
                   
        # temporal coherence loss
        temporal_coherence_loss = (s - s2).pow(2).sum(dim=1).mean()

        # InfoNCE loss
        predict_a = self.network(s, s2) #Size:(batch_size, a_dim)
        # ablation study use this a_mse_loss
        # a_mse_loss = (predict_a - a).pow(2).mean()
        
        infonce = InfoNCE(negative_mode='paired')
        query = predict_a
        positive_key = a
        negative_keys = negative_samples
        a_contrastive_loss = infonce(query, positive_key, negative_keys)

        # reward prediction loss
        predict_r = self.network.r_predictor(s, a).squeeze() #Size:(batch_size)
        r_mse_loss = (predict_r - r).pow(2).mean()
        
        # auxiliary loss
        aux_loss = r_mse_loss + a_contrastive_loss + temporal_coherence_loss
        # aux_loss = r_mse_loss + a_mse_loss + temporal_coherence_loss
        
        opt_dict = dict(opt_p=self.optimizer)
        info_dict = dict(RMseLoss=r_mse_loss.clone(), ACLoss=a_contrastive_loss.clone(), TCLoss=temporal_coherence_loss.clone(), AuxLoss=aux_loss.clone())
        # info_dict = dict(RMseLoss=r_mse_loss.clone(), AMseLoss=a_mse_loss.clone(), TCLoss=temporal_coherence_loss.clone(), AuxLoss=aux_loss.clone())
        return aux_loss, opt_dict, info_dict

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('RMseLoss', average_only=True)
        logger.log_tabular('ACLoss', average_only=True)
        # logger.log_tabular('AMseLoss', average_only=True)
        logger.log_tabular('TCLoss', average_only=True)
        logger.log_tabular('AuxLoss', average_only=True)
