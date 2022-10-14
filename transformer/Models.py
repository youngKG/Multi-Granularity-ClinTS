import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from transformer.Modules import Attention, ScaledDotProductAttention_bias
import sys
from einops import rearrange, repeat
from nwarp import Almtx

def get_len_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    seq = seq.ne(Constants.PAD)
    seq[:,0] = 1
    return seq.type(torch.float)

def get_attn_key_pad_mask_K(seq_k, seq_q, transpose=False):
    """ For masking out the padding part of key sequence. """
    # [B,L_q,K]
    if transpose:
        seq_q = rearrange(seq_q, 'b l k -> b k l 1')
        seq_k = rearrange(seq_k, 'b l k -> b k 1 l')
    else:
        seq_q = rearrange(seq_q, 'b k l -> b k l 1')
        seq_k = rearrange(seq_k, 'b k l -> b k 1 l')

    return torch.matmul(seq_q, seq_k).eq(Constants.PAD)

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    sz_b, len_s, type_num = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    
    subsequent_mask = rearrange(subsequent_mask, 'l l -> b k l l', b=sz_b, k=type_num)
    return subsequent_mask


def cal_delta_t(time_q, time_k):
    if time_q.dim() == 3: # [B,L,K]
        delta_t = rearrange(time_q, 'b l k -> b k l 1') - rearrange(time_k, 'b l -> b 1 1 l')
    else:
        delta_t = rearrange(time_q, 'b l -> b l 1') - rearrange(time_k, 'b l -> b 1 l')
    return delta_t 

class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(FFNN, self).__init__()

        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x

class Value_Encoder(nn.Module):
    def __init__(self, hid_units, output_dim, num_type):
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.num_type = num_type
        super(Value_Encoder, self).__init__()

        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask
    

class Event_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Event_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=Constants.PAD)

    def forward(self, event):
        # event = event * self.type_matrix
        event_emb = self.event_emb(event.long())
        return event_emb

class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else: # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1) # [B,L,1,D]
        out = torch.mul(out, self.k_map)
        return out * non_pad_mask # [B,L,K,D]

class MLP_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, num_types, hid_dim=16):
        super(MLP_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else: # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        # out1 = F.gelu(self.linear1(tt))
        tt = self.encoder(tt)
        tt = torch.mul(tt, self.k_map)
        return tt * non_pad_mask # [B,L,K,D]

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        
        super().__init__()
        self.opt = opt
        self.d_model = d_model
        self.embed_time = d_model

        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)]).to(opt.device)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        
        self.num_types = num_types

        self.enc_tau = opt.enc_tau
        self.a = nn.Parameter(torch.ones(1,num_types,1,1))
        self.b = nn.Parameter(torch.ones(1,num_types,1,1))
        self.sigma = nn.Parameter(torch.ones(1,num_types,1,1))
        self.pi = torch.tensor(math.pi)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, opt=opt)
            for _ in range(n_layers)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)
        
        self.w_t = nn.Linear(1, num_types, bias=False)
            
        if self.enc_tau == 'mlp':
            self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)
        elif self.enc_tau == 'tem':
            self.tau_encoder = Time_Encoder(self.embed_time, num_types)
            # self.tau_encoder = self.learn_time_embedding
            
        self.agg_attention = Attention_Aggregator(d_model)


    def forward(self, event_time, event_value, non_pad_mask, tau=None):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''

        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d') # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d') # [B,K,L,D]
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        event_emb = self.type_matrix * non_pad_mask
        event_emb = self.event_enc(event_emb) 
        event_emb = rearrange(event_emb, 'b l k d -> b k l d') # [B,K,L,D]

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')
        k_output = value_emb + tau_emb + event_emb + tem_enc_k
        
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l') # [B,K,L]

        for enc_layer in self.layer_stack:
            k_output, _, _ = enc_layer(
                k_output,
                non_pad_mask=non_pad_mask) 

        non_pad_mask = rearrange(non_pad_mask, 'b k l -> b k l 1') # [B,K,L]
        output = self.agg_attention(k_output, non_pad_mask) # [B,D]
        # k_output = rearrange(k_output, 'b k l d -> b l k d')
        return output
    
    
class Hie_Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, opt,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        
        super().__init__()
        self.opt = opt
        self.d_model = d_model
        self.embed_time = d_model

        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)]).to(opt.device)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        self.num_types = num_types
        
        
        if self.opt.task == "mor":
            self.new_l = 48
            width = 0.05
        elif self.opt.task == "wbm" or self.opt.task == "los" or self.opt.task == "decom":
            self.new_l = 24
            width = 0.1
        else:
            self.new_l = 6
            width = 0.2
            
        if opt.new_l > 0:
            self.new_l = opt.new_l
            if self.new_l == 48:
                width = 0.05
            elif self.new_l == 24:
                width = 0.1
            elif self.new_l == 12:
                width = 0.15
            elif self.new_l == 6:
                width = 0.2
            elif self.new_l == 4:
                width = 0.5
            
        self.time_split = [i for i in range(self.new_l+1)]
        self.enc_tau = opt.enc_tau

        self.low_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, opt=opt)
            for _ in range(n_layers)])
        
        self.high_layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, opt=opt)
            for _ in range(n_layers)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, num_types)
        
        self.w_t = nn.Linear(1, num_types, bias=False)

        if self.enc_tau == 'mlp':
            self.tau_encoder = MLP_Tau_Encoder(self.embed_time, num_types)
        elif self.enc_tau == 'tem':
            self.tau_encoder = Time_Encoder(self.embed_time, num_types)
            
        self.get_almtx = Almtx(opt, d_model, self.new_l, width=width, power=16, min_step=0.0001, max_step=0.9999)
        
        self.linear = nn.Linear(d_model*2, d_model)
        self.agg_attention = Attention_Aggregator(d_model)

    def aggregate_value(self, event_time, event_value, tau, non_pad_mask):
        b = event_value.shape[0]
        event_time = repeat(event_time, 'b l -> b l k', k=self.num_types)
        new_event_time = torch.zeros((b,self.new_l,self.num_types)).to(event_time.device)
        new_event_value = torch.zeros((b,self.new_l,self.num_types)).to(event_time.device)
        new_pad_mask = torch.zeros((b,self.new_l,self.num_types)).to(event_time.device)
        new_tau = torch.zeros((b,self.new_l,self.num_types)).to(event_time.device)
        
        for i in range(self.new_l):
            idx = (event_time.ge(i) & event_time.lt(i+1))
            total = torch.sum(idx, axis=1)
            total[total==0] = 1
            new_event_value[:,i,:] = torch.sum(event_value * idx, axis=1) / total
            new_event_time[:,i,:] = torch.sum(event_time * idx, axis=1) / total
            new_pad_mask[:,i,:] = torch.sum(non_pad_mask * idx, axis=1) / total
            new_tau[:,i,:] = torch.sum(tau * idx, axis=1) / total
            
        return new_event_value, new_event_time, new_tau, new_pad_mask
    
    def hour_aggregate(self, time_split, event_time, h0, non_pad_mask):
        # time_split: [new_l]
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]
        new_l = len(time_split)-1
        b, _, l, dim = h0.shape
        
        event_time_k = repeat(event_time, 'b l -> b k l', k=self.num_types)
        new_event_time = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        new_h0 = torch.zeros((b,self.num_types,new_l, dim)).to(h0.device)
        new_pad_mask = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        almat = torch.zeros((b,l,new_l)).to(h0.device) # [B,K,L]
        
        # for each time slot
        for i in range(len(time_split)-1):
            idx = (event_time_k.ge(time_split[i]) & event_time_k.lt(time_split[i+1])) # [B,K,L]
            total = torch.sum(idx, axis=-1) # [B,K]
            total[total==0] = 1
            
            tmp_h0 = h0 * idx.unsqueeze(-1) # [B,K,L,D]
            tmp_h0 = rearrange(tmp_h0, 'b k l d -> (b k) d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze() # [BK,D,1]
            new_h0[:,:,i,:] = rearrange(tmp_h0, '(b k) d -> b k d', b=b)
            almat[:,:,i] = (event_time.ge(time_split[i]) & event_time.lt(time_split[i+1])) # [B,L]

            new_event_time[:,:,i] = torch.sum(event_time_k * idx, axis=-1) / total
            new_pad_mask[:,:,i] = torch.sum(non_pad_mask * idx, axis=-1) / total
            
        return new_h0, new_event_time, new_pad_mask, almat
    
    def almat_aggregate(self, new_l, event_time, h0, non_pad_mask, test_bound=False):
        # K: the number of clusters
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]

        b, k, l, dim = h0.shape

        len_mask = get_len_pad_mask(event_time).to(h0.device)
        event_time = repeat(event_time, 'b l -> b k l', k=k)
        new_event_time = torch.zeros((b,k,new_l)).to(h0.device)
        new_h0 = torch.zeros((b,k, new_l, dim)).to(h0.device)
        new_pad_mask = torch.zeros((b,k,new_l)).to(h0.device)

        h = rearrange(h0, 'b k l d -> (b k) l d')
        len_mask = rearrange(non_pad_mask, 'b k l -> (b k) l')
        
        if test_bound:
            _, almat, m, mu = self.get_almtx(h, mask=len_mask, kernel='linear', test_bound=test_bound) #(batch_size, L), (batch_size, L, new_l)
        else:
            _, almat = self.get_almtx(h, mask=len_mask, kernel='linear') #(batch_size, L), (batch_size, L, new_l)
            
        almat = almat.to(h0.device)
        
        almat = rearrange(almat, '(b k) l s -> b k l s', k=k)

        for i in range(new_l):
            idx = almat[:,:,:,i]
            total = torch.sum(idx*non_pad_mask, axis=-1) # [B,K]
            total[total==0] = 1
            
            # weighted sum
            tmp_h0 = h0 * idx.unsqueeze(-1) # [B,K,L,D]

            # pooling
            tmp_h0 = rearrange(tmp_h0, 'b k l d -> (b k) d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze() # [BK,D]
            new_h0[:,:,i,:] = rearrange(tmp_h0, '(b k) d -> b k d', b=b)
            # sum
            # new_h0[:,:,i,:] = torch.sum(tmp_h0, dim=2)
            
            new_event_time[:,:,i] = torch.sum(event_time * idx, axis=-1) / total
            new_pad_mask[:,:,i] = torch.sum(non_pad_mask * idx, axis=-1) / total

        if test_bound:
            return new_h0, new_event_time, new_pad_mask, almat, m, mu
        else:
            return new_h0, new_event_time, new_pad_mask, almat

    def forward(self, event_time, event_value, non_pad_mask, tau=None, return_almat=False, test_bound=False):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''
        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d') # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d') # [B,K,L,D]
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        event_emb = self.type_matrix * non_pad_mask
        event_emb = self.event_enc(event_emb) 
        event_emb = rearrange(event_emb, 'b l k d -> b k l d') # [B,K,L,D]

        tau_emb = self.tau_encoder(tau, non_pad_mask)
        tau_emb = rearrange(tau_emb, 'b l k d -> b k l d')
        h0 = value_emb + tau_emb + event_emb + tem_enc_k
        
        # normalization
        # h0 = F.normalize(unorm_h0.float(),p=1,dim=-1)
        
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l') # [B,K,L]
        
        # high level
        
        if not self.opt.adpt:
            z0, _, new_pad_mask, almat = self.hour_aggregate(self.time_split, event_time, h0, non_pad_mask)
        elif test_bound:
            z0, _, new_pad_mask, almat, modes, mu = self.almat_aggregate(self.new_l, event_time, h0, non_pad_mask, test_bound=test_bound)
        else:
            z0, _, new_pad_mask, almat = self.almat_aggregate(self.new_l, event_time, h0, non_pad_mask, test_bound=test_bound)

        for enc_layer in self.low_layer_stack:
            enc_output, _, _ = enc_layer(
                h0,
                non_pad_mask=non_pad_mask) 

        # encode high level
        for high_enc_layer in self.high_layer_stack:
            new_h0, _, _ = high_enc_layer(
                z0, 
                non_pad_mask=new_pad_mask)
            
        # concatenation
        if not self.opt.adpt:
            new_h0 = torch.matmul(rearrange(new_h0, 'b k s d -> b k d s'), rearrange(almat, 'b l s -> b 1 s l')) # [B,K,D,L]
        else:
            new_h0 = torch.matmul(rearrange(new_h0, 'b k s d -> b k d s'), rearrange(almat, 'b k l s -> b k s l')) # [B,K,D,L]
        new_h0 = rearrange(new_h0, 'b k d l -> b k l d')
        enc_output = torch.cat([enc_output, new_h0], -1) 

        enc_output = self.linear(enc_output)
        output = self.agg_attention(enc_output, rearrange(non_pad_mask, 'b k l -> b k l 1'))

        unorm_h0 = h0
        if test_bound:
            return output, almat, unorm_h0, z0, modes, mu
        
        if return_almat:
            return output, almat
        else:
            return output


    
class Attention_Aggregator(nn.Module):
    def __init__(self, dim):
        super(Attention_Aggregator, self).__init__()
        self.attention_len = Attention(dim*2, dim)
        self.attention_type = Attention(dim*2, dim)

    def forward(self, ENCoutput, mask):
        """
        input: [B,K,L,D], mask: [B,K,L]
        """
        ENCoutput, _ = self.attention_len(ENCoutput, mask) # [B,K,D]
        ENCoutput, _ = self.attention_type(ENCoutput) # [B,D]
        return ENCoutput
    
class Classifier(nn.Module):

    def __init__(self, dim, type_num, cls_dim, activate=None):
        super(Classifier, self).__init__()
        # self.attention = Attention_Aggregator(dim)
        self.activate = activate
        self.linear1 = nn.Linear(dim, type_num)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(type_num, cls_dim)

    def forward(self, ENCoutput):
        """
        input: [B,L,K,D], mask: [B,L,K]
        """
        ENCoutput = self.linear1(ENCoutput)
        if self.activate:
            ENCoutput = self.sigmoid(ENCoutput)
        ENCoutput = self.linear2(ENCoutput)
        return ENCoutput
