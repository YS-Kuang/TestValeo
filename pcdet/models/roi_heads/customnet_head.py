# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:25:09 2022

@author: 54756
"""
import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
import numpy as np


class CustomNetHead(RoIHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, code_size, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)

        self.AoA_mat = np.load(r'/content/sample_data/Valeo/CalibrationTable.npy', allow_pickle=True).item()
        self.model_cfg = model_cfg

        self.radar_channels = [512]
        self.conv2d_kernel_size = (3, 3)
        self.pool_kernel_size_2d = (2, 2)
        self.max_pool_stride_2d = [2, 2]
        
        radar_inchannel = 256
        radar_modules = []
        for k in range(0, self.radar_channels.__len__()):
            radar_modules.extend([
                nn.Conv2d(radar_inchannel, self.radar_channels[k], self.conv2d_kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),
                nn.MaxPool2d(self.pool_kernel_size_2d, stride=self.max_pool_stride_2d, padding=0, dilation=1, return_indices=False, ceil_mode=False),
                nn.ReLU(),
                nn.BatchNorm2d(self.radar_channels[k])
                ])
            radar_inchannel = self.radar_channels[k]
        self.radar_layer = nn.Sequential(*radar_modules)
        
        
        pre_channel = 3072  #((512 + 128) * 5)
        
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=7,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def radar_processing(self, radar_features):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = radar_features.shape[0]
        doppler_indexes = np.array(self.AoA_mat['doppler_indexes'])
        window = torch.from_numpy(self.AoA_mat['H'][0]).float().to(device)
        CalibMat = torch.from_numpy(self.AoA_mat['Signal'][...,5]).to(device)
        MIMO_Spectrum = radar_features[:,:, doppler_indexes, :].reshape(batch_size, radar_features.shape[1] * radar_features.shape[2], -1)
        MIMO_Spectrum = torch.multiply(MIMO_Spectrum, window)
        MIMO_Spectrum = MIMO_Spectrum.transpose(1,2)
        radar_cubes = torch.abs(CalibMat@MIMO_Spectrum)
        radar_cubes = radar_cubes.reshape(batch_size, self.AoA_mat['Signal'].shape[0], radar_features.shape[1], radar_features.shape[2]) # (batch_size, 751, 512, 256)

        return radar_cubes

    def get_radar_features(self, radar_cubes, masks):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        r_axis_m = torch.from_numpy(self.AoA_mat['r_axis']).to(device)
        sin_azi_axis = torch.from_numpy(self.AoA_mat['sin_azi_axis']).to(device)
        doppler_features = []
        for batch_idx in range(radar_cubes.shape[0]):
            radar_cube = radar_cubes[batch_idx]
            mask = masks[batch_idx]
            all_range = torch.sqrt(mask[:,:,0]**2 + mask[:,:,1]**2)
            all_sin_azi = mask[:,:,1] / all_range
            all_range = all_range.view(-1,1)
            all_sin_azi = all_sin_azi.view(-1,1)
            all_range = torch.abs(all_range - r_axis_m).argmin(dim=1).view(-1,1)
            all_sin_azi = torch.abs(all_sin_azi - sin_azi_axis).argmin(dim=1).view(-1,1)
            doppler_feature = radar_cube[all_sin_azi, all_range, :].unsqueeze_(0)
            doppler_features.append(doppler_feature)
        doppler_features = torch.cat(doppler_features, dim=0)
        return doppler_features
        
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        if self.training:
          targets_dict = batch_dict.get('roi_targets_dict', None)
          if targets_dict is None:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features'] # (B, N, 5, 512)
            batch_dict['roi_masks'] = targets_dict['roi_masks'] # (B, N, 9, 256)
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        radar_cubes = self.radar_processing(batch_dict['radar_features'])
        doppler_features = self.get_radar_features(radar_cubes, batch_dict['roi_masks']) # (B, N, 9, 256)
        
        # radar modules
        doppler_features = doppler_features.view(-1, doppler_features.shape[2], 256) # (B * N, 9, 256)
        doppler_features = doppler_features.reshape(-1, 3, 3, 256)
        doppler_features = doppler_features.permute(0, 3, 1, 2).contiguous()
        features = self.radar_layer(doppler_features) # (B*N, 512, 1, 1)
        features = features.squeeze()
        
        # RoI aware pooling
        roi_features = batch_dict['roi_features'].view(-1, batch_dict['roi_features'].shape[2], 512)
        roi_features = roi_features.reshape(-1, 5*512)
        pooled_features = torch.cat((roi_features, features), dim=-1)
        
        pooled_features = pooled_features.reshape(-1, 1, 6*512).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict 
