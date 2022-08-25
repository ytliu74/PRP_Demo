import sys

sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from typing import List, Tuple, Sequence, Union, Optional
from wavenet590.modules.attention import (
    SelfAttention,
    PersonalizedAttention,
    HawkesAttention,
)
from wavenet590.modules.embed import PositionalEmbedding

from transformer.models.transformer_layer import (
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
)
from transformer.models.attn import FullAttention, AttentionLayer
from transformer.models.embed import DataEmbedding


class NewHawkesAttention(nn.Module):
    """Hawkes process for attending to contexts with query"""

    def __init__(self, device, dimensions, attention_type="general"):
        super(NewHawkesAttention, self).__init__()
        self.device = device
        if attention_type not in ["dot", "general"]:
            raise ValueError("Invalid attention type selected.")

        self.attention_type = attention_type
        if self.attention_type == "general":
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.ae = nn.Parameter(torch.FloatTensor([0.25]).to(device), requires_grad=True)
        self.ab = nn.Parameter(torch.FloatTensor([0.25]).to(device), requires_grad=True)

    def forward(self, query, context):
        batch_size, dimensions = query.size()
        seq_len = context.size(1)

        if self.attention_type == "general":
            query = self.linear_in(query)

        att_scores = torch.bmm(query.unsqueeze(1), context.transpose(1, 2).contiguous())
        att_scores = F.softmax(att_scores, dim=-1)
        mix = att_scores * (context.permute(0, 2, 1))

        # --- time dacay impact ---
        delta_t = (
            torch.flip(torch.arange(0, seq_len), [0])
            .type(torch.float32)
            .to(self.device)
        )
        delta_t = delta_t.repeat(batch_size, 1).reshape(batch_size, 1, seq_len)

        bt = torch.exp(-1 * self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2 + mix, -1)
        combined = torch.cat((mix, query), dim=-1)
        output = self.linear_out(combined)
        output = torch.tanh(output)
        return output


class TemporalModule(nn.Module):
    """时序模块，将时序数据处理成特征
    基本是一个类似WaveNet的结构，先进行Position Embedding，然后经过多层的dilate conv，并应用Hawkes Attention
    """

    def __init__(
        self,
        device: torch.device,
        in_dim: int,
        filter_channels: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 3, 4),
        dropout: float = 0.3,
        positional_embedding_d: int = 128,
        positional_embedding_max_len: int = 100,
    ):
        super(TemporalModule, self).__init__()
        self.dilations = dilations
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.hawks_att = nn.ModuleList()
        self.device = device
        self.start_conv = nn.Conv1d(
            in_channels=in_dim, out_channels=filter_channels, kernel_size=1
        )
        self.self_attention = SelfAttention(
            filter_channels, attention_type="dot", norm=True, drop=dropout
        )

        self.position_embedding = PositionalEmbedding(
            d_model=positional_embedding_d, max_len=positional_embedding_max_len
        )

        for i in range(len(self.dilations)):
            self.filter_convs.append(
                nn.Conv1d(
                    in_channels=filter_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    dilation=self.dilations[i],
                )
            )

            self.gate_convs.append(
                nn.Conv1d(
                    in_channels=filter_channels,
                    out_channels=filter_channels,
                    kernel_size=kernel_size,
                    dilation=self.dilations[i],
                )
            )

            self.norm.append(nn.BatchNorm1d(filter_channels))
            self.hawks_att.append(HawkesAttention(device, filter_channels))

    def forward(self, input: Tensor) -> Tensor:
        batch_size, num_nodes, num_days, num_features = input.shape
        input = input.reshape(-1, num_days, num_features)
        pe = self.position_embedding(input)
        input = input.permute(0, 2, 1)
        x = self.start_conv(input)

        # --- self attention ---
        x = x.permute(0, 2, 1) + pe
        x = x + self.self_attention(x, x, x)
        x = x.permute(0, 2, 1)

        stack_conv_out = []

        for i in range(len(self.dilations)):
            residual = x
            filter = self.filter_convs[i](residual)
            # filter = torch.tanh(filter)
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            x = x + residual[:, :, -x.size(2) :]
            # x_t = x.view(B, N, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)
            x = self.norm[i](x)  # .permute(0, 2, 1, 3).reshape(x.shape)

            # new content
            temp_x = x.permute(0, 2, 1)
            temp_x_past = torch.zeros(temp_x.shape, device=self.device)
            temp_x_past[:, 1:, :] = temp_x[:, :-1, :]
            temp_x_past[:, 0, :] = temp_x[:, 0, :]

            delta_x = temp_x - temp_x_past

            new_x = torch.cat((temp_x, delta_x), -1)
            permute_new_x = new_x.permute(0, 2, 1)

            # time-wise attention to readout this layer's feature sequence
            x_out = self.hawks_att[i](permute_new_x[:, :, -1], new_x)
            stack_conv_out.append(x_out)

        stack_conv_out = torch.stack(stack_conv_out, dim=1)
        # stack_conv_out = stack_conv_out.reshape(
        #     batch_size, num_nodes, stack_conv_out.shape[1], stack_conv_out.shape[2]
        # )

        return stack_conv_out


class FundTransformer(nn.Module):
    def __init__(
        self,
        c_in,
        device="cuda",
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.0,
        activation="gelu",
        petype=0,
        output_attention=False,
    ):
        super(FundTransformer, self).__init__()

        self.fund_embedding = DataEmbedding(c_in, d_model, dropout, petype)
        self.fund_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.hawks_attn = NewHawkesAttention(device, dimensions=d_model)

    def forward(self, fund_in, enc_self_mask=None):
        # shape_in: (bs, N, T, C)
        bs, n_stock, n_time, c = fund_in.shape
        fund_in = fund_in.view((-1, n_time, c))
        # shape_out: (bs, N, 1, C)
        fund_enc_in = self.fund_embedding(fund_in)
        fund_enc_out, _ = self.fund_encoder(fund_enc_in, attn_mask=enc_self_mask)

        fund_out = self.hawks_attn(fund_enc_out[:, -1, :].cuda(), fund_enc_out)
        fund_out = fund_out.view((bs * n_stock, 1, -1))
        return fund_out


class PlanA(nn.Module):
    def __init__(
        self,
        price_c_in,
        fund_c_in,
        c_out=1,
        d_model=128,
        n_heads=2,
        enc_layer=2,
        dec_layer=1,
        d_ff=256,
        dropout=0.3,
        activation="gelu",
        petype=0,
        num_nodes=590,
        stockid=30,
        type_att_channels=16,
        device="cuda",
    ):
        super(PlanA, self).__init__()

        self.d_model = d_model

        self.price_wavenet = TemporalModule(
            device,
            in_dim=price_c_in,
            filter_channels=d_model,
            positional_embedding_d=d_model,
            dropout=dropout,
        )

        self.fund_encoder = FundTransformer(
            fund_c_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=enc_layer,
            d_ff=d_ff,
            dropout=dropout,
            activation="gelu",
            petype=0,
        )

        self.hawks_attn = NewHawkesAttention(device, dimensions=d_model)

        self.stock_id_embedding = nn.Parameter(
            torch.randn(num_nodes, stockid, device=device), requires_grad=True
        )
        self.personalized_attn = PersonalizedAttention(
            id_dim=stockid, query_dim=type_att_channels, fea_dim=d_model
        )

        self.dec_embedding = DataEmbedding(price_c_in, d_model, dropout, petype)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False, attention_dropout=dropout, output_attention=False
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    n_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(dec_layer)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection_decoder = nn.Linear(d_model, c_out, bias=True)

    def forward(self, price_in, fund_in):
        bs = price_in.shape[0]
        stock_n = price_in.shape[1]

        price_wave_out = self.price_wavenet(price_in)

        fund_trans_out = self.fund_encoder(fund_in)
        fund_hawks_out = self.hawks_attn(
            fund_trans_out[:, -1, :], fund_trans_out
        ).unsqueeze(1)

        concat_feat = torch.cat((price_wave_out, fund_hawks_out), dim=1).view(
            (bs, stock_n, -1, self.d_model)
        )

        attn_out_once_list = []
        for concat_feat_once in concat_feat:
            attn_out_once = self.personalized_attn(
                self.stock_id_embedding, concat_feat_once
            )
            attn_out_once_list.append(attn_out_once)
        attn_out = torch.stack(attn_out_once_list).view((-1, 1, self.d_model))

        dec_in = self.dec_embedding(
            price_in[:, :, -1, :].view((-1, 1, price_in.shape[-1]))
        )

        dec_out = self.decoder(dec_in, attn_out)

        proj_out = self.projection_decoder(dec_out).view((bs, stock_n, 1))

        return proj_out
