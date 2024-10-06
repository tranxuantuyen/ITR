from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from .fuse_modules import BiAttentionBlock
from scipy.optimize import linear_sum_assignment
import numpy as np


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ItrSpatialTemporal(nn.Module):

    @configurable
    def __init__(
            self,
            in_channels,
            aux_loss,
            *,
            hidden_dim: int,
            num_frame_queries: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            enc_layers: int,
            dec_layers: int,
            enc_window_size: int,
            pre_norm: bool,
            enforce_input_project: bool,
            num_frames: int,
            num_classes: int,
            clip_last_layer_num: bool,
            conv_dim: int,
            mask_dim: int,
            sim_use_clip: list,
            use_sim: bool,
            cfg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # define Transformer decoder here
        self.cfg = cfg
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        self.cross_test = nn.ModuleList()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.clip_last_layer_num = clip_last_layer_num

        self.enc_layers = enc_layers
        self.window_size = enc_window_size
        self.sim_use_clip = sim_use_clip
        self.use_sim = use_sim
        self.aux_loss = aux_loss
        self.sptio_temp_encoder_layer = self.cfg.ITR.SPTIO_TEMP_ENCODER_LAYER

        self.enc_layers = enc_layers
        if enc_layers > 0:
            self.enc_self_attn = nn.ModuleList()
            # self.hierarchical_cross = nn.ModuleList()
            self.enc_ffn = nn.ModuleList()
            for _ in range(self.enc_layers):
                self.enc_self_attn.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    ),
                )
                self.enc_ffn.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
        if self.sptio_temp_encoder_layer > 0:
            self.spatial_encoder = nn.ModuleList()
            self.spatial_ffn = nn.ModuleList()
            if self.cfg.ITR.FUSE_VISION_TEXT == "concat":
                self.linear_concate_temporal = nn.ModuleList()
                self.linear_concate_spatial = nn.ModuleList()
                for _ in range(self.sptio_temp_encoder_layer):
                    self.linear_concate_spatial.append(nn.Linear(hidden_dim * 2, hidden_dim))
                    self.linear_concate_temporal.append(nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(self.sptio_temp_encoder_layer):
                self.spatial_encoder.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    ),
                )
                self.spatial_ffn.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            self.temp_encoder = nn.ModuleList()
            self.temp_ffn = nn.ModuleList()
            for _ in range(self.sptio_temp_encoder_layer):
                self.temp_encoder.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    ),
                )
                self.temp_ffn.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                ) 

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.cross_test.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.fusion_layers.append(
                BiAttentionBlock(
                    v_dim=hidden_dim,
                    l_dim=hidden_dim,
                    embed_dim=dim_feedforward // 2,
                    num_heads=self.num_heads // 2,
                    dropout=0.1,
                    drop_path=0.1,
                )
            )
        self.vita_mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.vita_mask_features)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj_dec = nn.Sequential()
        self.src_embed = nn.Identity()

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.class_embed = nn.Linear(hidden_dim * 2, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        if self.use_sim:
            self.sim_embed_frame = nn.Linear(hidden_dim, hidden_dim)
            if self.sim_use_clip:
                self.sim_embed_clip = nn.Linear(hidden_dim, hidden_dim)
        self.cross_motion = CrossAttentionLayer(d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.1,
                    normalize_before=pre_norm)
        self.cross_noun = CrossAttentionLayer(d_model=hidden_dim,
            nhead=nheads,
            dropout=0.1,
            normalize_before=pre_norm)
        self.cross_verb = CrossAttentionLayer(d_model=hidden_dim,
            nhead=nheads,
            dropout=0.1,
            normalize_before=pre_norm)
        self.noun_feat_init = nn.Embedding(1, hidden_dim)
        self.verb_feat_init = nn.Embedding(1, hidden_dim)
        if self.cfg.ITR.WEIGHT_RESUDIAL_PATH:
            self.weight_resudial_path = nn.Linear(hidden_dim, hidden_dim)
        if self.cfg.ITR.WEIGHT_RESUDIAL_IN_RNN:
            self.weight_resudial_RNN = nn.Linear(hidden_dim, hidden_dim)

    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels

        ret["hidden_dim"] = cfg.MODEL.VITA.HIDDEN_DIM
        ret["num_frame_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_queries"] = cfg.MODEL.VITA.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.VITA.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.VITA.DIM_FEEDFORWARD

        assert cfg.MODEL.VITA.DEC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.VITA.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.VITA.DEC_LAYERS
        ret["enc_window_size"] = cfg.MODEL.VITA.ENC_WINDOW_SIZE
        ret["pre_norm"] = cfg.MODEL.VITA.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.VITA.ENFORCE_INPUT_PROJ

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["clip_last_layer_num"] = cfg.MODEL.VITA.LAST_LAYER_NUM

        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["sim_use_clip"] = cfg.MODEL.VITA.SIM_USE_CLIP
        ret["use_sim"] = cfg.MODEL.VITA.SIM_WEIGHT > 0.0
        ret["cfg"] = cfg
        return ret

    def forward(self, frame_query, lang_feat, lang_mask, motion_feat=None, itr_feat=None, cal_procedure=None):
        """
        L: Number of Layers.
        B: Batch size.
        T: Temporal window size. Number of frames per video.
        C: Channel size.
        fQ: Number of frame-wise queries from IFC.
        cQ: Number of clip-wise queries to decode Q.
        """
        assert len(itr_feat) == 2
        if not self.training:
            frame_query = frame_query[[-1]]

        L, BT, fQ, C = frame_query.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L * B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query)  # T, fQ, LB, C

        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))  # _T, fQ, LB, C
            enc_mask = frame_query.new_ones(L * B, _T).bool()  # LB, _T
            enc_mask[:, :T] = False
        else:
            enc_mask = None
        noun_feat_init = self.noun_feat_init.weight.unsqueeze(1).repeat(1, B, 1)
        verb_feat_init = self.verb_feat_init.weight.unsqueeze(1).repeat(1, B, 1)
        noun_feat = [noun_feat_init + self.cross_noun(noun_feat_init, itr_.unsqueeze(1).repeat(1, B, 1)) for itr_ in itr_feat[0]]
        verb_feat = [verb_feat_init + self.cross_verb(verb_feat_init, itr_.unsqueeze(1).repeat(1, B, 1)) for itr_ in itr_feat[1]]
        itr_refine_feat = (noun_feat, verb_feat)

        frame_query = self.encode_frame_query(frame_query, enc_mask, cal_procedure, itr_refine_feat)
        frame_query = frame_query[:T].flatten(0, 1)  # TfQ, LB, C

        if self.use_sim:
            pred_fq_embed = self.sim_embed_frame(frame_query)  # TfQ, LB, C
            pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            pred_fq_embed = None

        src = self.src_embed(frame_query)  # TfQ, LB, C
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)  # TfQ, LB, C

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C
        output = self.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

        output_motion = self.cross_motion(output, motion_feat.unsqueeze(1).repeat(1, L, 1))
        output = output + 0.1 * output_motion
        decoder_outputs = []

        lang_feat_fusion = lang_feat.repeat(L, 1, 1).transpose(0, 1)
        lang_mask = lang_mask.repeat(L, 1)

        for i in range(self.num_layers):
            # attention: cross-attention first

            src, lang_feat_fusion = self.fusion_layers[i](
                v=src,
                l=lang_feat_fusion,
                attention_mask_v=None,
                attention_mask_l=~lang_mask.bool(),
            )

            output = self.transformer_cross_attention_layers[i](
                output, src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos, query_pos=query_embed
            )

            output = self.cross_test[i](
                output,
                lang_feat_fusion,
                memory_key_padding_mask=~lang_mask.bool()
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                dec_out = self.decoder_norm(output)  # cQ, LB, C
                dec_out = dec_out.transpose(0, 1)  # LB, cQ, C
                decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)  # D, L, B, cQ, C

        pred_cls = self.class_embed(decoder_outputs)
        pred_mask_embed = self.mask_embed(decoder_outputs)
        if self.use_sim and self.sim_use_clip:
            pred_cq_embed = self.sim_embed_clip(decoder_outputs)
        else:
            pred_cq_embed = [None] * self.num_layers

        lang_feat_mask = lang_feat_fusion.transpose(0, 1) * lang_mask.unsqueeze(-1)
        lang_feat_mean = torch.sum(lang_feat_mask, dim=1) / lang_mask.sum(dim=1, keepdim=True)
        out = {
            'pred_logits': pred_cls[-1],
            'pred_mask_embed': pred_mask_embed[-1],
            ' ': lang_feat_mean[-1], # lang_feat_mean
            'pred_fq_embed': pred_fq_embed,
            'pred_cq_embed': pred_cq_embed[-1],
            'aux_outputs': self._set_aux_loss(
                pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
            )
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(
            self, outputs_cls, outputs_mask_embed, outputs_cq_embed, outputs_fq_embed
    ):
        return [{"pred_logits": a, "pred_mask_embed": b, "pred_cq_embed": c, "pred_fq_embed": outputs_fq_embed}
                for a, b, c in zip(outputs_cls[:-1], outputs_mask_embed[:-1], outputs_cq_embed[:-1])]
    def np_vp_refinement(self,layer_idx, frame_query, step):
        text = step.lang_feat
        if step.label == 'NP':
            refined_query = frame_query.flatten(1, 2).permute(1, 0, 2) 
            refined_query = refined_query + text
            refined_query = self.spatial_encoder[layer_idx](refined_query)
            refined_query = self.spatial_ffn[layer_idx](refined_query)
            refined_query = refined_query.permute(1, 0, 2).view(frame_query.shape)
        else:
            refined_query = frame_query.flatten(1, 2)
            refined_query = refined_query + text
            refined_query = self.temp_encoder[layer_idx](refined_query)
            refined_query = self.temp_ffn[layer_idx](refined_query)
            refined_query = refined_query.view(frame_query.shape)
        return refined_query
        
    def spatial_temporal_transformer(self,layer_idx, frame_query, calculation_procedure, noun_verb_feature):
        if calculation_procedure is not None:
            allType = [ty.label for ty in calculation_procedure]
        else:
            allType = []
        if calculation_procedure is None and 'NP' in allType and 'VP' in allType:
            frame_query_dict = {}
            for step in calculation_procedure:
                if len(step.child) == 0:
                    frame_query_dict[step.id] = frame_query + self.np_vp_refinement(layer_idx, frame_query, step=step)
                else:
                    child_query = [frame_query_dict[child_id] for child_id in step.child]
                    child_query = sum(child_query) / len(child_query)
                    frame_query_dict[step.id] = self.np_vp_refinement(layer_idx, child_query, step=step)
            return frame_query_dict['0_0_0']

        else:
            recurrent_num = len(noun_verb_feature[0])
            all_noun, all_verb = noun_verb_feature
            for recurrent_idx in range(recurrent_num):
                noun, verb = all_noun[recurrent_idx], all_verb[recurrent_idx]
                return_shape = frame_query.shape
                spatial_query = frame_query.flatten(1, 2).permute(1, 0, 2) 
                if self.cfg.ITR.FUSE_VISION_TEXT == 'add':
                    spatial_query = spatial_query + noun
                elif self.cfg.ITR.FUSE_VISION_TEXT == 'concat':
                    noun = noun.repeat(spatial_query.shape[0], spatial_query.shape[1], 1)
                    spatial_query = torch.cat([spatial_query, noun], dim=-1)
                    spatial_query = self.linear_concate_spatial[layer_idx](spatial_query)
                else:
                    raise NotImplementedError
                spatial_query = self.spatial_encoder[layer_idx](spatial_query)
                spatial_query = self.spatial_ffn[layer_idx](spatial_query)
                spatial_query = spatial_query.permute(1, 0, 2).view(return_shape)

                temporal_query = spatial_query.flatten(1, 2)
                if self.cfg.ITR.FUSE_VISION_TEXT == 'add':
                    temporal_query = temporal_query + verb
                elif self.cfg.ITR.FUSE_VISION_TEXT == 'concat':
                    verb = verb.repeat(temporal_query.shape[0], temporal_query.shape[1], 1)
                    temporal_query = torch.cat([temporal_query, verb], dim=-1)
                    temporal_query = self.linear_concate_temporal[layer_idx](temporal_query)
                else:
                    raise NotImplementedError
                temporal_query = self.temp_encoder[layer_idx](temporal_query)
                temporal_query = self.temp_ffn[layer_idx](temporal_query)
                temporal_query = temporal_query.view(return_shape)
            if self.cfg.ITR.WEIGHT_RESUDIAL_IN_RNN:
                frame_query = self.weight_resudial_RNN(frame_query.flatten(0, 2)).reshape(return_shape)
            return temporal_query + frame_query

    def encode_frame_query(self, frame_query, attn_mask, cal_procedure, itr_refine_feat):
        """
        input shape (frame_query)   : T, fQ, LB, C
        output shape (frame_query)  : T, fQ, LB, C
        """

        # Not using window-based attention if self.window_size == 0.
        assert len(itr_refine_feat) == 2
        assert itr_refine_feat[0][0].shape == itr_refine_feat[1][0].shape
        if self.window_size == 0:
            return_shape = frame_query.shape  # T, fQ, LB, C
            frame_query = frame_query.flatten(0, 1)  # TfQ, LB, C

            for i in range(self.enc_layers):
                frame_query = self.enc_self_attn[i](frame_query)
                frame_query = self.enc_ffn[i](frame_query)

            frame_query = frame_query.view(return_shape)
            return frame_query
        # Using window-based attention if self.window_size > 0.
        else:
            T, fQ, LB, C = frame_query.shape
            W = self.window_size
            Nw = T // W
            half_W = int(ceil(W / 2))

            window_mask = attn_mask.view(LB * Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)

            _attn_mask = torch.roll(attn_mask, half_W, 1)
            _attn_mask = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)  # LB, Nw, W, W
            _attn_mask[:, 0] = _attn_mask[:, 0] | _attn_mask[:, 0].transpose(-2, -1)
            _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1)
            _attn_mask[:, 0, :half_W, half_W:] = True
            _attn_mask[:, 0, half_W:, :half_W] = True
            _attn_mask = _attn_mask.view(LB * Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB * Nw * self.num_heads, W * fQ, W * fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.enc_layers):
                if self.training or layer_idx % 2 == 0:
                    frame_query = self._window_attn(frame_query, window_mask, layer_idx, cal_procedure, itr_refine_feat)
                else:
                    frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx, cal_procedure, itr_refine_feat)
            return frame_query

    def _window_attn(self, frame_query, attn_mask, layer_idx, cal_procedure, itr_refine_feat):
        T, fQ, LB, C = frame_query.shape
        # LBN, WTfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W

        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1, 2, 3, 0, 4).reshape(W * fQ, LB * Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB * Nw, C)
        if self.cfg.ITR.WEIGHT_RESUDIAL_PATH:
            frame_query = self.weight_resudial_path(frame_query.flatten(0, 2)).reshape(T, fQ, LB, C)
            frame_query = frame_query + self.spatial_temporal_transformer(layer_idx, frame_query,cal_procedure, itr_refine_feat)
        else:
            frame_query = frame_query + self.spatial_temporal_transformer(layer_idx, frame_query,cal_procedure, itr_refine_feat)
            
        frame_query = frame_query.reshape(W * fQ, LB * Nw, C)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3, 0, 1, 2, 4).reshape(T, fQ, LB, C)

        return frame_query

    def _shift_window_attn(self, frame_query, attn_mask, layer_idx, cal_procedure, itr_refine_feat):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1, 2, 3, 0, 4).reshape(W * fQ, LB * Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = frame_query.reshape(W, fQ, LB * Nw, C)
        if self.cfg.ITR.WEIGHT_RESUDIAL_PATH:
            frame_query = self.weight_resudial_path(frame_query.flatten(0, 2)).reshape(T, fQ, LB, C)
            frame_query = frame_query + self.spatial_temporal_transformer(layer_idx, frame_query,cal_procedure, itr_refine_feat)
        else:
            frame_query = frame_query + self.spatial_temporal_transformer(layer_idx, frame_query,cal_procedure, itr_refine_feat)
        frame_query = frame_query.reshape(W * fQ, LB * Nw, C)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3, 0, 1, 2, 4).reshape(T, fQ, LB, C)
        frame_query = torch.roll(frame_query, -half_W, 0)
        return frame_query