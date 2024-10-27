import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_

from torch import Tensor
from typing import Optional

from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=8):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.sof = nn.Softmax(dim=2)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.linear3 = nn.Linear(d_model * 2, d_model)
        self.linear4 = nn.Linear(d_model , 2)
        self.linear5 = nn.Linear(d_model , d_model)


    def forward(self, input1, input2):
        # dif_as_kv
        fea = input1 +input2
        fea = self.activation(torch.cat([input1,input2],dim=2))
        att = self.linear3(fea)
        att = self.linear4(att)
        att = self.sof(att)
        att1 = att[:,:,0].unsqueeze(2)
        att2 = att[:,:,1].unsqueeze(2)
        fea = att1*input1+att2*input2 + input1+ input2
        fea = self.att(fea)
        input1 = self.att(input1)
        input2 = self.att(input2)
        output_1 = self.cross1(input1, fea) + self.cross1(input1,input2)  # (Q,K,V)
        output_2 = self.cross1(input2, fea) + self.cross2(input2,input1) # (Q,K,V)
        return output_1,output_2

    def cross1(self, input1,input2):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input1, input2, input2)  # (Q,K,V)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output
    
    def cross2(self, input1,input2):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input1, input2, input2)  # (Q,K,V)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output
        
    def att(self, input1):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input1, input1, input1)  # (Q,K,V)
        output = input1 + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3,stride=1,padding=1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                # nn.LayerNorm(int(outchannel / 1),dim=1)
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)

class spatial_att(nn.Module):
    def __init__(self, feature_dim):
        super(spatial_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv= nn.Conv2d(feature_dim, feature_dim, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))
    def forward(self,x):
        FL = self.up(self.avg_pool(x))
        FH = x - FL
        max_out, _ = torch.max(FH, dim=1, keepdim=True)
        min_out, _ = torch.min(FH, dim=1, keepdim=True)
        return self.conv(x*self.sigmoid(max_out)*self.alpha + x*self.sigmoid(min_out)*(1-self.alpha))


class semantic_att(nn.Module):
    def __init__(self, feature_dim,reduction=16):
        super(semantic_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv= nn.Conv2d(feature_dim, feature_dim, 1, padding=0, bias=True)
        # feature channel downscale and upscale --> channel weight
        self.sem_weight = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.sem_weight(y)
        return self.conv(x * y)





class CVFF(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=1):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(CVFF, self).__init__()
        self.d_model = d_model

        # n_layers = 3
        print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers
        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))

        self.projection1 = nn.Conv2d(2048, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(2048, d_model, kernel_size=1)
        self.linearprojection1 = nn.Linear(2048, d_model)
        
        self.linear = nn.Linear(d_model , d_model * 2)
        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        self.resblock = nn.ModuleList([resblock(d_model*2, d_model*2) for i in range(n_layers)])
        self.LN = nn.ModuleList([nn.LayerNorm(d_model*2) for i in range(n_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 64 2048 14 14  img_feat2 34 2048 14 14
       
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat2.size(0)
        feature_dim = img_feat2.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)
        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)

        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)
        # (h, w, d_model)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                     1)  # (batch, d_model, h, w)

        output1 = (self.projection1(img_feat1)+position_embedding).reshape(batch, 512, -1).permute(2, 0, 1)
        output2 = (self.projection2(img_feat2)+position_embedding).reshape(batch, 512, -1).permute(2, 0, 1)

        output1_list = list()
        output2_list = list()
        for l in self.transformer:
            output1, output2 = l(output1, output2)
            output1_list.append(output1)
            output2_list.append(output2)

        # MBF
        i = 0
        output = torch.zeros((196,batch,self.d_model*2)).to(device)

        for res in self.resblock:
            input = torch.cat([output1_list[i],output2_list[i]],dim=-1)
            output = output + input
            output = output.permute(1, 2, 0).view(batch, self.d_model*2, 14, 14)
            output = res(output)
            output = output.view(batch, self.d_model*2,-1).permute(2, 0, 1)
            output = self.LN[i](output)
            i=i+1
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.embedding_1D = nn.Embedding(27, int(d_model))

    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        x = x + self.embedding_1D(torch.arange(27, device=device).to(device)).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)
        
class PositionalEncoding1(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding1, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(14, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        x = x + self.embedding_1D(torch.arange(14, device=device).to(device)).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        self.multihead_attn4 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)


        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor,   tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # cross self-attention

        enc_att, att_weight = self._mha_block2((tgt),
                                              memory, memory_mask,
                                               memory_key_padding_mask)

       
        

        x = self.norm2(tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight
    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout3(x),att_weight
    def _mha_block3(self, x: Tensor, mem1: Tensor,mem2: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn3(x, mem1, mem2,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x)

    def _mha_block4(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn4(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout5(x),att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)


class Mesh_TransformerDecoderLayer2(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer2, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn4 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.sig = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None
                ) -> Tensor:
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention


        enc_att1, att_weight = self._mha_block1(self_att_tgt,
                                               memory, memory_mask,
                                               memory_key_padding_mask)
        
        #enc_att2, att_weight = self._mha_block2((self_att_tgt),
        #                                       tnom, tnom_mask,
        #                                       tnom_key_padding_mask)
    

        branch1 = self.norm1(self.sig(enc_att1)*enc_att1+enc_att1)
        #branch2 = self.norm1(self.sig(enc_att2)*enc_att2+enc_att1)
        branch2 = self.norm2(self.sig(self_att_tgt)*self_att_tgt+self_att_tgt)
        #branch3 = self.norm3(self.sig(enc_att2)*enc_att2+enc_att2)

        # branch11, att_weight = self._mha_block4((branch1+self_att_tgt),
        #                                       memory, memory_mask,
        #                                       memory_key_padding_mask)
        # branch22, att_weight = self._mha_block4((branch2+self_att_tgt),
        #                                      memory, memory_mask,
        #                                     memory_key_padding_mask)

        x = self.norm2(self_att_tgt +branch1+branch2)
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block1(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                            attn_mask=attn_mask,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=True)
        return self.dropout2(x), att_weight

    def _mha_block2(self, x: Tensor, mem: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn2(x, mem, mem,
                                             attn_mask=attn_mask,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=True)
        return self.dropout3(x), att_weight

    def _mha_block3(self, x: Tensor, mem1: Tensor, mem2: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn3(x, mem1, mem2,
                                             attn_mask=attn_mask,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=True)
        return self.dropout4(x)

    def _mha_block4(self, x: Tensor, mem: Tensor,
                    attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn4(x, mem, mem,
                                             attn_mask=attn_mask,
                                             key_padding_mask=key_padding_mask,
                                             need_weights=True)
        return self.dropout5(x), att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)

class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=",n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding

        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        decoder_layer1 = Mesh_TransformerDecoderLayer2(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)


                                                   
        self.fc1 = nn.Linear(512, feature_dim)
        self.fc2 = nn.Linear(512, feature_dim)
        #self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.transformer1 = nn.TransformerDecoder(decoder_layer1, n_layers)

        self.position_encoding = PositionalEncoding(feature_dim)
        self.position_encoding1 = PositionalEncoding1(feature_dim)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.wdc1 = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory,encoded_captions, encoded_words, caption_lengths, word_lengths):
        """
        :param memory: image feature (S, batch, feature_dim)
        :param tgt: target sequence (length, batch)
        :param sentence_index: sentence index of each token in target sequence (length, batch)
        """
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        word = encoded_words.permute(1, 0)
        word_length = word.size(0)

        mask1 = (torch.triu(torch.ones(word_length, word_length)) == 1).transpose(0, 1)
        mask1 = mask1.float().masked_fill(mask1 == 0, float('-inf')).masked_fill(mask1 == 1, float(0.0))
        mask1 = mask1.to(device)


        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        word_embedding = self.vocab_embedding(word)
        word_embedding = self.position_encoding1(word_embedding)  # (length, batch, feature_dim)

        pred_word = self.transformer(word_embedding, memory,  tgt_mask=mask1)  # (length, batch, feature_dim)
        pred = self.transformer1(tgt_embedding, memory, tgt_mask=mask)  # (length, batch, feature_dim)
        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
        pred_word = self.wdc1(self.dropout(pred_word))  # (length, batch, vocab_size)

        pred = pred.permute(1, 0, 2)
        pred_word = pred_word.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        word_lengths, sort_ind1 = word_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_words = encoded_words[sort_ind1]
        pred_word = pred_word[sort_ind1]
        decode2_lengths = (word_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind, pred_word, encoded_words,  decode2_lengths, sort_ind1


