def update_config(args):
    args.lr = 0.0003
    args.label_num = 16
    args.batchsize = 512
    args.embed_dim = 300
    args.kernel_sizes = "3,4,5"
    args.kernel_num = 100
    args.static = False
    args.dropout = 0.5
    args.epochs = 20
    args.clip = 1.0
    args.alpha = 0.0
    args.emb = None
    args.best = True
