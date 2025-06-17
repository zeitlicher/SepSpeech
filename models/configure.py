
def configure():
    config={}

    # position encoder parameters
    config['max_len'] = 80000 # samples

    # encoder/decoder parameters
    config['channels'] = 256
    config['kernel_size']=16
    config['stride']=8

    # chnuk layer
    config['chunk_size']=250

    # sepformer paramters
    config['d_model'] = 256
    config['nhead'] = 8
    config['dim_feedforward'] = 1024
    config['layer_norm_eps'] = 1.e-8
    config['num_layers'] = 4 # default=8

    # separator parameters
    config['num_sepformer_layers'] = 1 # default = 2

    # speaker networks
    config['n_speakers'] = 256

    # model paramters
    config['dropout'] = 0.1

    return config
