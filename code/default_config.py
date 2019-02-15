# -*- coding: UTF-8 -*-
def get_default_config():
    args = lambda: None

    # Data path
    args.datadir = '../data'
    # Incorporate validation data into vocabulary
    args.incorp_val = False
    # Random seed
    args.seed = None
    # Initial learning rate
    args.lr = 0.1
    # Shuffle the data every epoch
    args.shuffle = True
    # Store model of the best epoch
    args.best = False
    # Fix the embeddings 
    args.static = False
    # Use existing word embeddings
    args.emb = None
    return args
