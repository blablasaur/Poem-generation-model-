import numpy as np
import timeit
import torch
from sklearn.metrics import accuracy_score


class Config():
    def __init__(self,
                 vocab_size,
                 device = None,
                 dtype=None,
                 context_length = 256,
                 n_layers = 6,
                 h_dim =256,
                 n_heads = 8,
                 mlp_exp= 4,
                 batch_size = 32,
                 lr = 3 * 1e-4 ,
                 optimizer= "AdamW",
                 dropout = 0.1,
                 training_steps = 30_000 ,
                 ):
        self.context_length = context_length
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.n_heads = n_heads 
        self.mlp_exp = mlp_exp
        self.batch_size = batch_size
        self.lr = lr 
        self.optimizer = optimizer
        self.dropout = dropout
        self.training_steps = training_steps
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype 


def reconstruct_poem(gen_tokens,STANZA_BREAK,vocab):
    decoded_tokens = []
    for t in gen_tokens:
        d = vocab[t]
        decoded_tokens.append(d)
        
    poem = "".join(decoded_tokens)
    poem.replace(STANZA_BREAK,"\n\n")
    return poem


def tokenize(txt: str, w2i , BOS):
    spl = list(txt.lower())
    return [w2i[BOS]] + [w2i[c] for c in spl if c in w2i.keys() ]


def char_acc(Y_true : torch.Tensor , logits : torch.Tensor):
    Y_pred = logits.argmax(dim=-2)
    return accuracy_score(Y_true,Y_pred)
    


