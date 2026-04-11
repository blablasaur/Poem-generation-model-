"""
This file contains TransformerEncoder, TransformerDecoder and Transformer
copied almost verbatim from the PyTorch codebase. The only change is that
the "fast path" logic in the TransformerEncoder is removed. And the src/key/memory
padding mask is removed.

"""

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor    
from src.utils import Config
from src.td_layer import TransformerDecoderLayer
import math

# We use this for exact parity with the PyTorch implementation, having the same init
# for every layer might not be necessary.

#In TransformerDecoderLayer, get rid of mha (cross ) attention block only keep the self attention , for causal just set the forward arg tgt_is_causal to true in the forward (memory doesn't concern RAM but more  the result of the encoder in basic transformer imp )  to 

class GPT(nn.Module):
    def __init__(self,config: Config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size,config.h_dim)
        self.pos = SinusoidalPositionalEncoding(config.h_dim,context_length = config.context_length)
        self.layers = nn.ModuleList((TransformerDecoderLayer(d_model = config.h_dim, dim_feedforward= config.h_dim * config.mlp_exp,device = config.device, dtype = config.dtype,nhead=config.n_heads) for _ in range(config.n_layers)))
        self.output = nn.Linear(config.h_dim,config.vocab_size)


    def forward(self, idxs: Tensor):
        x = self.embeddings(idxs)
        pos_enc = self.pos(x)
        x += pos_enc
        for layer  in self.layers:
            x = layer(x,tgt_is_causal = True)
        logits = self.output(x)
        return logits
    

    def get_prob_distribution(self, logits):
        return torch.nn.functional.softmax(logits,dim=-1)

    def generate_1tk(self,tokens, top_k, temperature = 1.0):
        logits = self.forward(tokens)[-1,:]

        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = self.get_prob_distribution(logits)
        gen_token = torch.multinomial(probs,1)
        return gen_token
    
    def generate(self, context,EOS_index,  context_length=256, top_k = 30):
        for i in range(context_length - len(context)):
            gen_token = self.generate_1tk(context, top_k)
            context = torch.cat([context, gen_token],dim=0)
            if gen_token == EOS_index:
                return context
        return context 

        




class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, context_length):
        super().__init__()

        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        seq_len = x.size(-2)
        if len(x.shape) < 3:
            return self.pe[0,:seq_len, :]
        else:
            return self.pe[:,:seq_len, :]
        
