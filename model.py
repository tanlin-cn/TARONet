import torch
import torch.nn.functional as F
from torch import nn

# from torchvision.ops import nms
import matplotlib.pyplot as plt
import math
import copy

from functools import partial
from torchmetrics.classification import BinaryJaccardIndex, F1Score, BinaryPrecisionRecallCurve

import lightning.pytorch as pl
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d

import wandb
import pprint
import torchvision

import SAC

import vmamba
import decoder

import torch
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

class TopoDecoder(nn.Module):
    def __init__(self, config):
        super(TopoDecoder, self).__init__()
        self.config = config
        self.decoder = decoder.Decoder_topo(config=config, filters=[96, 192, 384, 768])

    def forward(self, feature_maps):
        return self.decoder(feature_maps)
class BilinearSampler(nn.Module):
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points):
        """
        Args:
            feature_maps (Tensor): The input feature tensor of shape [B, D, H, W].
            sample_points (Tensor): The 2D sample points of shape [B, N_points, 2],
                                    each point in the range [-1, 1], format (x, y).
        Returns:
            Tensor: Sampled feature vectors of shapes [B, N_points, D].
        """
        # feature_maps = feature_maps[3]
        B, D, H, W = feature_maps.shape
        _, N_points, _ = sample_points.shape

        # normalize cooridinates to (-1, 1) for grid_sample
        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        
        # sample_points from [B, N_points, 2] to [B, N_points, 1, 2] for grid_sample
        sample_points = sample_points.unsqueeze(2)
        
        # Use grid_sample for bilinear sampling. Align_corners set to False to use -1 to 1 grid space.
        # [B, D, N_points, 1]
        sampled_features = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        
        # sampled_features is [B, N_points, D]
        sampled_features = sampled_features.squeeze(dim=-1).permute(0, 2, 1)
        return sampled_features

class TopoNet(nn.Module):
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config

        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3

        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 2, self.hidden_dim)

        # Create Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True  # Input format is [batch size, sequence length, features]
        )
        
        # Stack the Transformer Encoder Layers
        if self.config.TOPONET_VERSION != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, pairs, pairs_valid):
        # points: [B, N_points, 2]
        # point_features: [B, N_points, D]
        # pairs: [B, N_samples, N_pairs, 2]
        # pairs_valid: [B, N_samples, N_pairs]
        
        point_features = F.relu(self.feature_proj(point_features))
        # gathers pairs
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        # Use advanced indexing to fetch the corresponding feature vectors
        # [B, N_samples * N_pairs, D]
        src_features = point_features[batch_indices, pairs[:, :, 0].to(torch.long)]
        tgt_features = point_features[batch_indices, pairs[:, :, 1].to(torch.long)]
        # [B, N_samples * N_pairs, 2]
        src_points = points[batch_indices, pairs[:, :, 0].to(torch.long)]
        tgt_points = points[batch_indices, pairs[:, :, 1].to(torch.long)]
        offset = tgt_points - src_points

        ## ablation study
        # [B, N_samples * N_pairs, 2D + 2]
        if self.config.TOPONET_VERSION == 'no_tgt_features':
            pair_features = torch.concat([src_features, torch.zeros_like(tgt_features), offset], dim=2)
        if self.config.TOPONET_VERSION == 'no_offset':
            pair_features = torch.concat([src_features, tgt_features, torch.zeros_like(offset)], dim=2)
        else:
            pair_features = torch.concat([src_features, tgt_features, offset], dim=2)
        
        
        # [B, N_samples * N_pairs, D]
        pair_features = F.relu(self.pair_proj(pair_features))
        
        # attn applies within each local graph sample
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        # valid->not a padding
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)

        # [B * N_samples, 1]
        #### flips mask for all-invalid pairs to prevent NaN
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)

        padding_mask = ~pairs_valid
        
        ## ablation study
        if self.config.TOPONET_VERSION != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        
        ## Seems like at inference time, the returned n_pairs heres might be less - it's the
        # max num of valid pairs across all samples in the batch
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)

        # [B, N_samples, N_pairs, 1]
        logits = self.output_proj(pair_features)

        scores = torch.sigmoid(logits)

        return logits, scores


class TARORoad(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        prompt_embed_dim = 768
        image_size = config.PATCH_SIZE
        self.image_size = image_size

        patch_size = 4
        image_embedding_size = image_size // patch_size

        encoder_output_dim = prompt_embed_dim

        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

        self.image_encoder = vmamba.Backbone_VSSM(
          img_size=image_size,
          out_indices=(0, 1, 2, 3),
          pretrained=None,
          # copied from classification/configs/vssm/vssm_tiny_224.yaml
          dims=96,
          # depths=(2, 2, 5, 2),
          depths=(2, 2, 8, 2),
          ssm_d_state=1,
          ssm_dt_rank="auto",
          # ssm_ratio=2.0,
          ssm_ratio=1.0,
          ssm_conv=3,
          ssm_conv_bias=False,
          forward_type="v05_noz",  # v3_noz,
          mlp_ratio=4.0,
          downsample_version="v3",
          patchembed_version="v2",
          drop_path_rate=0.2,
          norm_layer="ln2d",
          )
        
        activation = nn.GELU
        self.map_decoder = nn.Sequential(
          nn.ConvTranspose2d(encoder_output_dim, 384, kernel_size=2, stride=2),
          LayerNorm2d(384),
          activation(),
          nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
          activation(),
          nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
          activation(),
          nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
          activation(),
          nn.ConvTranspose2d(48, 2, kernel_size=2, stride=2),
        )
        
        #### TOPONet
        self.sac = SAC.SAC(encoder_output_dim, r=4)
        self.topo_decoder = TopoDecoder(config)
        self.bilinear_sampler = BilinearSampler(config)
        self.topo_net = TopoNet(config, 48)

        #### Losses
        self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        # self.topo_criterion = loss.dice_bce_loss_topo()
        #### Metrics
        self.keypoint_iou = BinaryJaccardIndex(threshold=0.5)
        self.road_iou = BinaryJaccardIndex(threshold=0.5)
        self.topo_f1 = F1Score(task='binary', threshold=0.5, ignore_index=-1)
        # testing only, not used in training
        self.keypoint_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.road_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)
        self.topo_pr_curve = BinaryPrecisionRecallCurve(ignore_index=-1)

        with open(config.SAM_CKPT_PATH, "rb") as f:
          ckpt_state_dict = torch.load(f)

          matched_names = []
          mismatch_names = []
          state_dict_to_load = {}
          for k, v in self.named_parameters():
              if "backbone."+k[14:] in ckpt_state_dict['state_dict'] and v.shape == ckpt_state_dict['state_dict']["backbone."+k[14:]].shape:
                  matched_names.append(k)
                  state_dict_to_load[k] = ckpt_state_dict['state_dict']["backbone."+k[14:]]
              else:
                  mismatch_names.append(k)
          print("###### Matched params ######")
          pprint.pprint(matched_names)
          print("###### Mismatched params ######")
          pprint.pprint(mismatch_names)

          self.matched_param_names = set(matched_names)
          self.mismatch_param_names = set(mismatch_names)
          self.load_state_dict(state_dict_to_load, strict=False)

        
    def resize_sam_pos_embed(self, state_dict, image_size, patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

    
    def forward(self, rgb, graph_points, pairs, valid):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        # VM+TL： image_embeddings is a list, outputs of four stage
        # VM+simple(4 upsample): self.image_encoder return a list
        # SAM+simple: image_embeddings is a tensor

        image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        image_embeddings[3] = self.sac(image_embeddings[3])
        mask_logits = self.map_decoder(image_embeddings[3])
        mask_scores = torch.sigmoid(mask_logits)
        
        ## Predicts local topology
        image_embeddings = self.topo_decoder(image_embeddings)
        point_features = self.bilinear_sampler(image_embeddings, graph_points)

        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        
        
        # [B, H, W, 2]
        mask_logits = mask_logits.permute(0, 2, 3, 1)
        mask_scores = mask_scores.permute(0, 2, 3, 1)

        return mask_logits, mask_scores, topo_logits, topo_scores
    
    def infer_masks_and_img_features(self, rgb):
        # rgb: [B, H, W, C]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        x = rgb.permute(0, 3, 1, 2)
        # [B, C, H, W]
        x = (x - self.pixel_mean) / self.pixel_std
        # [B, D, h, w]
        if self.config.VM and not self.config.TL:
            image_embeddings = self.image_encoder(x)[3]
        else:
            image_embeddings = self.image_encoder(x)
        # mask_logits, mask_scores: [B, 2, H, W]
        image_embeddings[3] = self.saff(image_embeddings[3])
        mask_logits = self.map_decoder(image_embeddings[3])
        mask_scores = torch.sigmoid(mask_logits)
        
        # [B, H, W, 2]
        mask_scores = mask_scores.permute(0, 2, 3, 1)

        return mask_scores, image_embeddings
    

    def infer_toponet(self, config, image_embeddings, graph_points, pairs, valid):
        # image_embeddings: [B, D, h, w]
        # graph_points: [B, N_points, 2]
        # pairs: [B, N_samples, N_pairs, 2]
        # valid: [B, N_samples, N_pairs]

        ## Predicts local topology
        image_embeddings = self.topo_decoder(image_embeddings)
        point_features = self.bilinear_sampler(image_embeddings, graph_points)

        # [B, N_sample, N_pair, 1]
        topo_logits, topo_scores = self.topo_net(graph_points, point_features, pairs, valid)
        return topo_scores


    def training_step(self, batch, batch_idx):
        # opt_mask, opt_topo = self.optimizers()
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)

        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        ### DEBUG NAN


        topo_loss *= topo_loss_mask.unsqueeze(-1)
        # topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()

        loss = mask_loss + topo_loss


        self.log('train_mask_loss', mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)


        mask_loss = self.mask_criterion(mask_logits, gt_masks)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)

        # [B, N_samples, N_pairs, 1]
        topo_loss = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))

        topo_loss *= topo_loss_mask.unsqueeze(-1)
        # topo_loss = torch.nansum(torch.nansum(topo_loss) / topo_loss_mask.sum())
        topo_loss = topo_loss.sum() / topo_loss_mask.sum()
        val_loss = mask_loss + topo_loss
        self.log('val_mask_loss', mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num, :, :]
            viz_pred_keypoint = mask_scores[:max_viz_num, :, :, 0]
            viz_pred_road = mask_scores[:max_viz_num, :, :, 1]
            viz_gt_keypoint = keypoint_mask[:max_viz_num, ...]
            viz_gt_road = road_mask[:max_viz_num, ...]
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road']
            data = [[wandb.Image(x.cpu().numpy()) for x in row] for row in list(zip(viz_rgb, viz_gt_keypoint, viz_gt_road, viz_pred_keypoint, viz_pred_road))]
            self.logger.log_table(key='viz_table', columns=columns, data=data)

        self.keypoint_iou.update(mask_scores[..., 0], keypoint_mask)
        self.road_iou.update(mask_scores[..., 1], road_mask)

        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_f1.update(topo_scores, topo_gt.unsqueeze(-1))
        

    def on_validation_epoch_end(self):
        keypoint_iou = self.keypoint_iou.compute()
        road_iou = self.road_iou.compute()
        topo_f1 = self.topo_f1.compute()
        self.log("keypoint_iou", keypoint_iou)
        self.log("road_iou", road_iou)
        self.log("topo_f1", topo_f1)
        self.keypoint_iou.reset()
        self.road_iou.reset()
        self.topo_f1.reset()

    def test_step(self, batch, batch_idx):
        # masks: [B, H, W]
        rgb, keypoint_mask, road_mask = batch['rgb'], batch['keypoint_mask'], batch['road_mask']
        graph_points, pairs, valid = batch['graph_points'], batch['pairs'], batch['valid']

        # masks: [B, H, W, 2] topo: [B, N_samples, N_pairs, 1]
        mask_logits, mask_scores, topo_logits, topo_scores = self(rgb, graph_points, pairs, valid)

        topo_gt, topo_loss_mask = batch['connected'].to(torch.int32), valid.to(torch.float32)

        self.keypoint_pr_curve.update(mask_scores[..., 0], keypoint_mask.to(torch.int32))
        self.road_pr_curve.update(mask_scores[..., 1], road_mask.to(torch.int32))

        valid = valid.to(torch.int32)
        topo_gt = (1 - valid) * -1 + valid * topo_gt
        self.topo_pr_curve.update(topo_scores, topo_gt.unsqueeze(-1).to(torch.int32))

    def on_test_end(self):
        def find_best_threshold(pr_curve_metric, category):
            print(f'======= {category} ======')   
            precision, recall, thresholds = pr_curve_metric.compute()
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_threshold_index = torch.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_index]
            best_precision = precision[best_threshold_index]
            best_recall = recall[best_threshold_index]
            best_f1 = f1_scores[best_threshold_index]
            print(f'Best threshold {best_threshold}, P={best_precision} R={best_recall} F1={best_f1}')
        
        print('======= Finding best thresholds ======')
        find_best_threshold(self.keypoint_pr_curve, 'keypoint')
        find_best_threshold(self.road_pr_curve, 'road')
        find_best_threshold(self.topo_pr_curve, 'topo')


    def configure_optimizers(self):
        param_dicts = []

        encoder_params = {
            'params': [p for k, p in self.image_encoder.named_parameters() if 'image_encoder.'+k in self.matched_param_names],
            'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
        }
        param_dicts.append(encoder_params)
        encoder_params = {
            'params': [p for k, p in self.image_encoder.named_parameters() if
                        'image_encoder.' + k in self.mismatch_param_names],
            'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
        }
        param_dicts.append(encoder_params)

        decoder_params = [{
            'params': [p for p in self.map_decoder.parameters()],
            'lr': self.config.BASE_LR * 0.1
        }]
        param_dicts += decoder_params


        topo_decoder_params = [{
            'params': [p for p in self.topo_decoder.parameters()],
            'lr': self.config.BASE_LR * 0.1
        }]
        param_dicts += topo_decoder_params

        topo_net_params = [{
            'params': [p for p in self.topo_net.parameters()],
            'lr': self.config.BASE_LR
        }]
        param_dicts += topo_net_params
        
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')

        optimizer = torch.optim.Adam(param_dicts, lr=self.config.BASE_LR)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9], gamma=0.1)
        return [{'optimizer': optimizer, 'lr_scheduler': step_lr}]


