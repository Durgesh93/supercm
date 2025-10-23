from .default import shared_config,dataset_config,alg_config,ce_config,cecm_config,labels_config


def get_config(key):
    setting = {}
    setting.update(shared_config)

    components = key.split('_')
    dataset     = components[0]
    base        = components[1]
    ce_cecm     = components[2]
    labels      = int(components[3])

    setting.update(dataset_config[dataset])
    setting.update(alg_config[base])
    setting.update(labels_config[labels])

    if ce_cecm == 'ce':
        setting.update(ce_config[base])
    elif ce_cecm == 'cecm':
        setting.update(cecm_config[base]) 
    
    return setting


def update_args(setting,args):
    setting['gamma']       = args.gamma
    setting['beta']        = args.beta
    setting['lr']          = args.lr
    setting['T']           = args.T
    setting['set_cenmode'] = args.set_cenmode
    return setting
